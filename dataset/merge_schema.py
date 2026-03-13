import json
import os
from tqdm import tqdm
from collections import Counter, defaultdict




def combine_splits(input_paths: list[str], output_path: str):
    with open(output_path, "w") as outfile:
        # Iterate over each input file path provided
        for input_path in input_paths:
            input_path = os.path.join("/home/banlujie/data/synsql_5k/train", input_path)
            print(f"Processing file: {input_path}")
            try:
                with open(input_path, "r") as infile:
                    for line in infile:
                        try:
                            data = json.loads(line)
                            outfile.write(json.dumps(data) + '\n')
                        except json.JSONDecodeError as e:
                            print(f"Could not process line in {input_path}: {line.strip()} - Error: {e}")
            except FileNotFoundError:
                print(f"Error: Input file not found at {input_path}")
    return


def find_duplicates_in_train_json(train_file_path: str):
    """
    分析 train.json 文件，找出其中重複的問題。
    """
    print(f"🚀 開始分析 {train_file_path} 內的重複問題...")

    if not os.path.exists(train_file_path):
        print(f"錯誤: 找不到文件 {train_file_path}。")
        return

    with open(train_file_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 用於儲存從「正規化問題」到「原始數據記錄列表」的映射
    normalized_q_to_records = defaultdict(list)

    # 遍歷所有數據，進行正規化並分類
    for item in tqdm(train_data, desc="正在讀取和分類"):
        question = item.get("question")
        if question:
            normalized_question = question.strip().lower()
            normalized_q_to_records[normalized_question].append(item)

    # 找出所有重複項
    duplicate_records = {}
    for normalized_q, records in normalized_q_to_records.items():
        if len(records) > 1:
            duplicate_records[normalized_q] = records

    # --- 診斷報告 ---
    print("\n--- 診斷報告 ---")
    if not duplicate_records:
        print(f"✅ 在 {len(train_data)} 條記錄中未發現重複的問題。")
    else:
        print(f"⚠️ 在 {len(train_data)} 條記錄中發現 {len(duplicate_records)} 組重複的問題！\n")

        count = 0
        for normalized_q, records in duplicate_records.items():
            count += 1
            print(f"--- 重複組 {count} ---")
            print(f"正規化後的問題: \"{normalized_q}\" (共出現 {len(records)} 次)")

            for record in records:
                db_id = record.get('db_id', 'N/A')
                original_q = record.get('question', 'N/A')
                print(f"  - db_id: {db_id}")
                print(f"    原始問題: \"{original_q}\"")
            print("-" * (len(f"--- 重複組 {count} ---")) + "\n")


def load_and_normalize_train_questions(train_file_path: str) -> set:
    """
    從 train.json 加載所有問題，並進行正規化處理。
    返回一個包含所有正規化後問題的集合。
    """
    print(f"正在從 {train_file_path} 加載並正規化問題...")
    question_set = set()
    try:
        with open(train_file_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
    except FileNotFoundError:
        print(f"錯誤: 找不到 {train_file_path}。")
        return None

    for item in tqdm(train_data, desc="處理 train.json"):
        question = item.get("question")
        if question:
            # 正規化：轉換為小寫並去除首尾空白
            normalized_question = question.strip().lower()
            question_set.add(normalized_question)

    print(f"✅ train.json 處理完畢，找到 {len(question_set)} 個獨立的正規化後的問題。")
    return question_set


def find_unmatched_questions(responses_file: str, train_questions_set: set):
    """
    遍歷響應文件，找出無法在訓練問題集合中匹配到的問題。
    """
    print(f"\n🚀 開始在 {responses_file} 中查找不匹配的問題...")

    if not os.path.exists(responses_file):
        print(f"錯誤: 找不到 {responses_file}。")
        return

    missing_questions = []

    with open(responses_file, 'r', encoding='utf-8') as f:
        # 假設您的 sct_batch1_full.json 是 jsonl 格式
        # 如果不是，您需要相應地修改讀取方式
        for line in tqdm(f, desc="掃描響應文件"):
            try:
                response_data = json.loads(line.strip())
                original_question = response_data.get("question")
                if not original_question:
                    continue

                # 對響應文件中的問題進行【同樣的】正規化
                normalized_question = original_question.strip().lower()

                # 檢查正規化後的問題是否存在於集合中
                if normalized_question not in train_questions_set:
                    # 如果不存在，我們將【原始的】問題加入列表以便檢查
                    missing_questions.append(original_question)
            except json.JSONDecodeError:
                tqdm.write("  [警告] 發現一個無法解析的 JSON 行，已跳過。")

    print("\n--- 診斷報告 ---")
    if not missing_questions:
        print("✅ 恭喜！所有問題都能在 train.json 中找到匹配項。")
    else:
        print(f"⚠️ 發現 {len(missing_questions)} 條無法匹配的問題，列表如下：")
        for i, q in enumerate(missing_questions, 1):
            print(f"{i}. \"{q}\"")
        print("\n請仔細檢查以上問題的文本，與 train.json 中的對應項進行比較，找出細微的字符差異。")


def load_json_file(filepath: str) -> dict:
    """加載一個標準的 JSON 文件。"""
    print(f"正在加載 JSON 文件: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"錯誤: 找不到文件 {filepath}。")
        return None


def load_jsonl_file(filepath: str) -> list:
    """加載一個 JSON Lines (.jsonl) 文件。"""
    print(f"正在加載 JSONL 文件: {filepath}")
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    except FileNotFoundError:
        print(f"錯誤: 找不到文件 {filepath}。")
        return None


def save_to_jsonl(data: list, filepath: str):
    """將數據列表保存為 JSON Lines (.jsonl) 文件。"""
    print(f"正在將 {len(data)} 條記錄保存至: {filepath}")
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("✅ 保存完畢。")


def main():
    """主執行函數，整合 schema 到 response 數據中。"""

    # --- 1. 定義文件路徑 ---
    # train_file = "/home/banlujie/data/synsql_5k/train.json"  # 包含 (question, db_id) 對應關係
    # schemas_file = "/home/banlujie/data/synsql_5k/database_schemas.jsonl"  # 包含 (db_id, schema_string) 對應關係
    # responses_file = "/home/banlujie/data/synsql_5k/train/scm_cot_batch1_full.json"  # 包含 (question, gemini_response)
    # output_file = "/home/banlujie/data/synsql_5k/train/enriched_scm_cot_batch4_full.jsonl"  # 最終輸出的合併文件
    input_files = ['enriched_scm_cot_batch1_full.jsonl','enriched_scm_cot_batch2_full.jsonl',
                   'enriched_scm_cot_batch3_full.jsonl','enriched_scm_cot_batch4_full.jsonl']
    output_file = "/home/banlujie/data/synsql_5k/train/enriched_scm_cot_all_full.jsonl"
    combine_splits(input_files, output_file)
    # --- 2. 加載所有需要的數據 ---
    # train_data = load_json_file(train_file)
    # schemas = load_json_file(schemas_file)
    # responses = load_jsonl_file(responses_file)
    # find_duplicates_in_train_json(train_file)
    # train_questions = load_and_normalize_train_questions(train_file)
    # if train_questions is not None:
    #     find_unmatched_questions(responses_file, train_questions)
    # if not all([train_data, schemas, responses]):
    #     print("缺少必要的輸入文件，程式終止。")
    #     return
    #
    # # --- 3. 建立 question -> db_id 的查找表 (橋樑) ---
    # print("正在建立 question -> db_id 的查找表...")
    # question_to_dbid_map = {
    #     item['question']: item['db_id']
    #     for item in tqdm(train_data, desc="掃描 train.json")
    # }
    # print(f"✅ 查找表建立完畢，包含 {len(question_to_dbid_map)} 條對應。")
    #
    # # --- 4. 遍歷響應數據，進行查找與合併 ---
    # enriched_data = []
    # missing_count = 0
    # for response_item in tqdm(responses, desc="整合數據"):
    #     question = response_item.get("question")
    #     if not question:
    #         continue
    #
    #     # 第一步查找: question -> db_id
    #     db_id = question_to_dbid_map.get(question)
    #     if not db_id:
    #         missing_count += 1
    #         tqdm.write(f"  [警告] 在 train.json 中找不到問題 '{question[:30]}...' 的 db_id。")
    #         continue
    #
    #     # 第二步查找: db_id -> schema_string
    #     schema_string = schemas.get(db_id)
    #     if not schema_string:
    #         missing_count += 1
    #         tqdm.write(f"  [警告] 在 schemas.json 中找不到 db_id '{db_id}' 的 Schema。")
    #         continue
    #
    #     # 成功找到，合併數據
    #     # 我們將 response_item 的所有內容複製過來，再添加新字段
    #     new_entry = {
    #         **response_item,
    #         "db_id": db_id,
    #         "schema": schema_string
    #     }
    #     enriched_data.append(new_entry)
    #
    # if missing_count > 0:
    #     print(f"\n⚠️ 處理完成，有 {missing_count} 條記錄因無法找到對應關係而被跳過。")
    #
    # # --- 5. 保存整合後的數據 ---
    # save_to_jsonl(enriched_data, output_file)


if __name__ == "__main__":
    # 為了讓腳本能直接運行，您可以先創建一些假的輸入文件
    # if not os.path.exists("train.json"):
    #     dummy_train = [{"question": "Who is the top scorer?", "db_id": "sports_db", "table_names_original": [],
    #                     "column_names_original": []}]
    #     with open("train.json", 'w') as f: json.dump(dummy_train, f)
    # if not os.path.exists("schemas.json"):
    #     with open("schemas.json", 'w') as f: json.dump({"sports_db": "CREATE TABLE players (...);"}, f)
    # if not os.path.exists("sct_batch1_full.json"):
    #     with open("sct_batch1_full.json", 'w') as f: f.write(
    #         json.dumps({"question": "Who is the top scorer?", "gemini_response": "..."}) + '\n')

    main()
