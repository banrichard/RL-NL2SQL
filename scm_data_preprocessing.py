#!/usr/bin/env python3
"""
将多个输入文件中的 response 提取并合并到一个 JSONL 文件中的脚本
"""
import argparse
import json
import os.path
import re
from copy import deepcopy
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset.scm_dataset import parse_schema_for_metadata


def initialize_empty_intents() -> dict:
    """返回一個空的意圖狀態字典。"""
    return {
        "select": [], "from": [], "join": [], "where": [],
        "group_by": [], "order_by": None, "aggregate": [], "having": [], "limit": None,
    }


def find_subsequence_indices(main_list: list, sub_list: list) -> int:
    """在主列表中查找子列表的起始索引。"""
    for i in range(len(main_list) - len(sub_list) + 1):
        if main_list[i:i + len(sub_list)] == sub_list:
            return i
    return -1  # 如果找不到


def parse_schema_to_columns(schema_string: str) -> dict:
    """
    從 CREATE TABLE 字符串中解析出 "table.column" 格式的列名列表。
    """
    if not schema_string:
        return {"tables": [], "columns": []}

    tables = []
    columns = []

    # 穩健的的正則表達式，用於匹配整個 CREATE TABLE 語句
    table_definitions = re.findall(
        r"CREATE TABLE\s+[`\"']?(\w+)[`\"']?\s*\((.*?)\)\s*;?",
        schema_string,
        re.DOTALL | re.IGNORECASE
    )

    for table_name, table_content in table_definitions:
        tables.append(table_name)

        # 逐行解析欄位，這是最穩健的方法
        lines = table_content.strip().split('\n')
        for line in lines:
            line = line.strip()
            # 精準匹配一行的開頭是否為一個被引號 "..." 包裹的欄位名
            col_match = re.match(r"[`\"'](\w+)[`\"']", line)

            if col_match:
                col_name = col_match.group(1)
                # 過濾掉可能是 PRIMARY KEY, FOREIGN KEY 等的關鍵字行
                if col_name.upper() not in ["PRIMARY", "FOREIGN", "CONSTRAINT"]:
                    columns.append(f"{table_name}.{col_name}")
    return {"tables": tables, "columns": columns}


def build_dataset_from_source(source_file: str, output_file: str, task_type: str, tokenizer: str):
    """
    【最終修正版】從您的程式碼版本修改而來，修正了 tokenizer 處理和邏輯重複問題。
    """
    print(f"🚀 開始為 '{task_type}' 任務構建數據集，從 {source_file}...")

    # 斷言，確保傳入的是 Fast Tokenizer
    base_path = "/home/banlujie/data/synsql_5k/train"
    source_file_path = os.path.join(base_path, source_file)
    output_file_path = os.path.join(base_path, output_file)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    if not os.path.exists(source_file_path):
        print(f"錯誤: 輸入文件 '{source_file_path}' 不存在。")
        return

    with open(source_file_path, 'r', encoding='utf-8') as f_in, \
            open(output_file_path, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(tqdm(f_in, desc=f"處理數據 ({task_type})"), 1):
            try:
                data = json.loads(line.strip())

                scm_eval = data.get('scm_evaluation')
                if not scm_eval or 'step_wise_evaluation' not in scm_eval or 'final_parsed_intents' not in scm_eval:
                    continue

                question = data['question']
                db_id = data['db_id']
                schema_str = data['schema']

                parsed_schema = parse_schema_to_columns(schema_str)
                schema_tables = parsed_schema['tables']
                full_schema_columns = parsed_schema['columns']
                schema_columns_str_for_input = " [SEP] ".join(full_schema_columns)

                step_evaluations = scm_eval['step_wise_evaluation']
                final_intents = scm_eval['final_parsed_intents']

                cot_history = ""
                current_intent_state = {key: [] if isinstance(val, list) else None for key, val in
                                        final_intents.items()}

                for step_data in step_evaluations:
                    step_description = step_data['description']

                    # --- 簡化邏輯：先確定 input_text 和 labels ---
                    if task_type == 'analyzer':
                        current_cot_history = cot_history + step_description + " "
                        input_text = f"[CLS] {question} [SEP] {schema_columns_str_for_input} [SEP] {current_cot_history.strip()} [SEP]"

                        focused_intents = step_data['focused_intent']
                        for intent in focused_intents:
                            intent_key = intent.lower().replace(' ', '_')
                            if intent_key in current_intent_state and intent_key in final_intents:
                                current_intent_state[intent_key] = final_intents[intent_key]

                        labels = {
                            "sql_intents": deepcopy(current_intent_state),
                            "quality_scores": {"convergence": step_data['convergence_score'],
                                               "efficiency": step_data['efficiency_score']},
                            "focused_intents": focused_intents
                        }

                    elif task_type == 'predictor':
                        # predictor 的輸入是上一步的歷史
                        input_text = f"[CLS] {question} [SEP] {schema_columns_str_for_input} [SEP] {cot_history.strip()} [SEP] [MASK] [SEP]"
                        labels = {"next_focused_intent": step_data['focused_intent']}

                    else:
                        continue

                    # --- 統一的 Tokenizer 呼叫和索引計算 ---
                    encoding = tokenizer(
                        input_text,
                        truncation=True,
                        return_offsets_mapping=True,
                        max_length=4096  # 確保與您的模型一致
                    )

                    # 【核心修正】: 不使用 return_tensors，直接獲取單層列表
                    offset_mapping = encoding.pop("offset_mapping", [])

                    column_char_indices = [input_text.find(col) for col in full_schema_columns]

                    column_token_indices = []
                    for char_start_index in column_char_indices:
                        token_index = 0
                        found = False
                        if char_start_index != -1 and offset_mapping:
                            # 防禦性檢查
                            for i, offset in enumerate(offset_mapping):
                                if isinstance(offset, (list, tuple)) and len(offset) == 2:
                                    start, end = offset
                                    if start <= char_start_index < end:
                                        token_index = i
                                        found = True
                                        break
                        column_token_indices.append(token_index if found else 0)

                    training_sample = {
                        "db_id": db_id, "question": question, "input_text": input_text,
                        "schema_columns": full_schema_columns,
                        "schema_tables": schema_tables,
                        "column_token_indices": column_token_indices,
                        "labels": labels
                    }
                    f_out.write(json.dumps(training_sample, ensure_ascii=False) + '\n')

                    # --- 唯一的歷史更新點 ---
                    cot_history += step_description + " "

            except Exception as e:
                tqdm.write(f"  [嚴重錯誤] 處理第 {line_num} 行時發生未預期錯誤: {type(e).__name__} - {e}")

    print(f"🎉 '{task_type}' 數據集構建完成！已保存至 {output_file}")


def find_unique_intent_keys(responses_file: str) -> set:
    """
    掃描數據文件，找出 "final_parsed_intents" 中所有出現過的獨立鍵。

    Args:
        responses_file: 整合了 Schema 和 Gemini 響應的 .jsonl 文件路徑。

    Returns:
        一個包含所有獨立意圖鍵的集合 (set)。
    """
    print(f"🚀 開始掃描文件 {responses_file} 以查找所有意圖類型...")

    if not os.path.exists(responses_file):
        print(f"錯誤: 找不到文件 {responses_file}。")
        return set()

    # 使用集合 (set) 來自動處理重複項
    all_intent_keys = set()

    with open(responses_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="正在分析意圖"):
            try:
                data = json.loads(line.strip())

                # 獲取 'final_parsed_intents' 字典
                # 為了穩健性，我們假設它可能在頂層，也可能在 'gemini_response' 鍵下
                intents_dict = None
                if 'final_parsed_intents' in data:
                    intents_dict = data['final_parsed_intents']
                elif 'scm_evaluation' in data and 'final_parsed_intents' in data['scm_evaluation']:
                    intents_dict = data['scm_evaluation']['final_parsed_intents']

                if intents_dict and isinstance(intents_dict, dict):
                    # 將該字典的所有鍵添加到我們的集合中
                    all_intent_keys.update(intents_dict.keys())

            except (json.JSONDecodeError, KeyError) as e:
                tqdm.write(f"  [警告] 處理某行時出錯，已跳過。錯誤: {e}")

    return all_intent_keys


def extract_schema_from_prompt(prompt_content: str) -> str:
    """
    從 prompt 文本中提取 Database Schema 區塊。

    Args:
        prompt_content: 包含 "Database Schema:" 的完整 prompt 字符串。

    Returns:
        提取出的 CREATE TABLE 語句字符串，如果找不到則返回 None。
    """
    # 使用正則表達式來定位 Schema 區塊
    # re.DOTALL 讓 '.' 可以匹配換行符
    match = re.search(r"Database Schema:(.*?)External Knowledge:", prompt_content, re.DOTALL | re.IGNORECASE)

    if match:
        # group(1) 獲取第一個捕獲組的內容，並去除首尾空白
        return match.group(1).strip()
    return None


def process_dataset_and_extract_schemas(input_file: str, output_file: str):
    """
    讀取 train.json，提取所有 db_id 的 schema，並保存到一個新文件。
    """
    print(f"🚀 開始從 {input_file} 提取 Schema...")

    if not os.path.exists(input_file):
        print(f"錯誤: 輸入文件 '{input_file}' 不存在。")
        return

    schemas_dict = {}

    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for item in tqdm(dataset, desc="正在處理數據"):
        db_id = item.get("db_id")
        prompt_list = item.get("prompt")

        if not db_id or not prompt_list:
            tqdm.write(f"  [警告] 找到一條缺少 'db_id' 或 'prompt' 的記錄，已跳過。")
            continue

        # prompt 是一個列表，我們通常處理第一個元素
        if isinstance(prompt_list, list) and len(prompt_list) > 0:
            content = prompt_list[0].get("content")
            if content:
                schema_string = extract_schema_from_prompt(content)
                if schema_string:
                    # 如果同一個 db_id 出現多次，後面的會覆蓋前面的
                    # 這通常是期望的行為，因為同一個 db_id 的 schema 應該相同
                    schemas_dict[db_id] = schema_string
                else:
                    tqdm.write(f"  [警告] 在 db_id '{db_id}' 的 prompt 中未找到 Schema 區塊。")

    print(f"\n💾 提取完成，共找到 {len(schemas_dict)} 個獨立的 Schema。正在保存至 {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(schemas_dict, f_out, indent=4, ensure_ascii=False)

    print(f"🎉 全部完成！Schema 已成功保存至 {output_file}")


def process_files(input_paths, output_path):
    """
    Reads multiple input JSONL files, extracts and cleans the 'text' field,
    and writes the inner JSON objects to a single output JSONL file.
    """
    # Open the single output file in write mode

    with open(output_path, "w") as outfile:
        # Iterate over each input file path provided
        for input_path in input_paths:
            input_path = os.path.join("/home/banlujie/data/synsql_5k", input_path)
            print(f"Processing file: {input_path}")
            try:
                with open(input_path, "r") as infile:
                    for line in infile:
                        try:
                            data = json.loads(line)
                            if 'response' in data and 'candidates' in data['response']:
                                for candidate in data['response']['candidates']:
                                    for part in candidate['content']['parts']:
                                        text_content = part['text']
                                        # Clean the string by removing markdown fences and whitespace
                                        cleaned_text = text_content.strip()
                                        if cleaned_text.startswith("```json"):
                                            cleaned_text = cleaned_text[7:]
                                        if cleaned_text.endswith("```"):
                                            cleaned_text = cleaned_text[:-3]

                                        # Parse the cleaned string into a JSON object
                                        inner_json_obj = json.loads(cleaned_text.strip())

                                        # Write the JSON object as a string to the output file
                                        outfile.write(json.dumps(inner_json_obj) + '\n')
                        except json.JSONDecodeError as e:
                            print(f"Could not process line in {input_path}: {line.strip()} - Error: {e}")
            except FileNotFoundError:
                print(f"Error: Input file not found at {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract responses from multiple JSON files into a single JSONL file.")
    # Use nargs='+' to accept one or more input file paths
    # parser.add_argument("--input_paths", type=str, nargs='+', required=True, help="Paths to the input files.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--task", type=str, choices=['analyzer', 'predictor'], required=False,
                        help="Type of task for dataset construction.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file.")
    parser.add_argument('--tokenizer_name', type=str, default="allenai/longformer-base-4096",
                        help="用於計算 token indices 的 tokenizer 名稱")
    args = parser.parse_args()
    build_dataset_from_source(args.input_path, args.output_path, args.task, args.tokenizer_name)
    # process_files(args.input_paths, args.output_path)
    # process_dataset_and_extract_schemas("/home/banlujie/data/synsql_5k/train.json", "/home/banlujie/data/synsql_5k/database_schemas.jsonl")bu
    #

    # print(f"Successfully created '{args.output_path}' with data from all input files.")
