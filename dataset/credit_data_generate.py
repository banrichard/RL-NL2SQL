# build_credit_dataset.py
import json
import os
from tqdm import tqdm
import sqlparse
import random
import time
from pebble import ProcessPool
from concurrent.futures import TimeoutError,as_completed
import re
from utils import execute_sql, get_detailed_score


# --- 1. 輔助函式 ---

def create_counterfactual_sql(tokens: list, index_to_mask: int) -> str:
    """
    建立反事實 SQL，這裡用 '[MASK]' 來遮擋指定位置的 token。
    """
    counterfactual_tokens = tokens[:index_to_mask] + ["[MASK]"] + tokens[index_to_mask + 1:]
    return " ".join(counterfactual_tokens)


def generate_clause_counterfactuals_with_parser(sql: str) -> dict:
    counterfactuals = {}

    # 1. 定義我們關心的主要子句關鍵字
    CLAUSE_KEYWORDS_TO_ANALYZE = {"SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "WITH"}

    try:
        parsed = sqlparse.parse(sql)[0]
        tokens = parsed.tokens
    except IndexError:
        return {}

    # 2. 遍歷所有頂層的 token 和 token group
    for i, token in enumerate(tokens):
        # 忽略空白符
        if token.is_whitespace:
            continue

        # 3. 檢查 token 是否是我們關心的子句
        # 獲取元件的第一個詞作為代表關鍵字
        component_keyword = str(token).split()[0].upper()

        if component_keyword in CLAUSE_KEYWORDS_TO_ANALYZE:
            # 建立一個移除了當前元件的新 token 列表
            counterfactual_tokens = tokens[:i] + tokens[i + 1:]
            counterfactual_sql = "".join(str(t) for t in counterfactual_tokens).strip()

            if counterfactual_sql:
                # 使用標準化的關鍵字作為 key
                counterfactuals[component_keyword] = counterfactual_sql

    return counterfactuals


def generate_clause_counterfactuals(sql: str) -> dict:
    """
    為一個 SQL 查詢生成所有可能的「移除子句」的反事實版本。
    """
    counterfactuals = {}
    clauses_to_remove = ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"]

    for clause in clauses_to_remove:
        # 使用不分大小寫的正規表示式來尋找並移除子句
        # 這會移除從子句關鍵字到下一個子句關鍵字或結尾的所有內容
        pattern = rf"\b{clause}\b.*?(?=\b(SELECT|FROM|WHERE|GROUP BY|HAVING|ORDER BY|LIMIT)\b|$)"
        match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)

        if match:
            # 建立移除了該子句的新 SQL
            counterfactual_sql = sql[:match.start()] + sql[match.end():]
            counterfactual_sql = " ".join(counterfactual_sql.split())  # 清理多餘空格
            counterfactuals[clause] = counterfactual_sql

    return counterfactuals


# --- 2. Ray 並行任務 ---

def process_clause_credit_task(task_data: dict, db_root_path: str):
    """
    對單一的 SQL 進行子句級別的信用分配計算。
    """
    chosen_sql = task_data['chosen']
    db_id = task_data['db_id']
    db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")

    # a. 計算基線獎勵
    ground_truth_result = execute_sql(db_path, chosen_sql)
    if isinstance(ground_truth_result, str) and "Error" in ground_truth_result:
        return None  # 標準答案有問題，跳過

    R_total = 2.0

    # b. 生成所有子句的反事實版本
    counterfactual_sqls = generate_clause_counterfactuals_with_parser(chosen_sql)

    if not counterfactual_sqls:
        return None  # 沒有可供分析的子句

    # c. 為每個被移除的子句計算信用值
    clause_credits = {}
    for clause, cf_sql in counterfactual_sqls.items():
        R_counterfactual = get_detailed_score(db_path, cf_sql, ground_truth_result)['score']
        credit = R_total - R_counterfactual
        clause_credits[f"{clause.replace(' ', '_')}"] = credit

    if not clause_credits:
        return None

    return {
        "prompt": task_data['prompt'],
        "db_id": db_id,
        "chosen_sql": chosen_sql,
        "clause_credits": clause_credits
    }


def postprocess_for_model_training(
        input_path: str = "clause_credit_dataset.jsonl",
        output_path: str = "credit_model_training_data.jsonl",
):
    """
    Converts detailed, variable-length credit data into a structured
    token-to-credit map for model training.
    """
    print(f"Converting '{input_path}' to a flexible model training format...")
    CLAUSE_ORDER = ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "WITH"]

    print(f"正在將 '{input_path}' 轉換為模型訓練格式...")
    print(f"將遵循以下子句順序: {CLAUSE_ORDER}")

    processed_count = 0
    with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="正在處理資料"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = data.get("prompt")
            chosen_sql = data.get("chosen_sql")
            clause_credits = data.get("clause_credits", {})

            if not prompt or not chosen_sql:
                continue

            # 2. 建立目標信用向量
            # 初始化一個長度與 CLAUSE_ORDER 相同，值全為 0 的向量
            target_credits = [0.0] * len(CLAUSE_ORDER)

            # 遍歷原始的 clause_credits 字典
            for i, clause in enumerate(CLAUSE_ORDER):
                # 檢查這個標準子句是否存在於從檔案讀取的 clause_credits 字典中
                if clause in clause_credits:
                    # 如果存在，就在向量的對應位置填入信用值
                    target_credits[i] = clause_credits[clause]

            # 3. 建立最終的訓練樣本
            training_sample = {
                # 輸入：將 prompt 和 SQL 拼接，用特殊符號分隔
                "prompt": f"{prompt} <SQL> {chosen_sql}",
                # 標籤：固定長度的信用向量
                "target_credits": target_credits
            }

            f_out.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
            processed_count += 1

    print("\n" + "=" * 50)
    print("Flexible post-processing complete!")
    print(f"Processed {processed_count} entries.")
    print(f"Final training file saved to: '{output_path}'")
    print("=" * 50)


def rename_key_in_jsonl(
        input_path: str = "credit_model_training_data.jsonl",
        output_path: str = "credit_model_training_data_renamed.jsonl",
        old_key: str = "input_text",
        new_key: str = "prompt"
):
    """
    逐行讀取一個 .jsonl 檔案，並將指定的金鑰名稱更改為新名稱。
    """
    print(f"正在讀取檔案 '{input_path}'...")
    print(f"準備將金鑰 '{old_key}' 更名為 '{new_key}'...")

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
                open(output_path, 'w', encoding='utf-8') as f_out:

            # 為了顯示進度條，我們先計算總行數
            total_lines = sum(1 for line in open(input_path, 'r', encoding='utf-8'))
            f_in.seek(0)  # 回到檔案開頭

            for line in tqdm(f_in, total=total_lines, desc="正在更名"):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 忽略格式錯誤的行

                # 檢查舊金鑰是否存在
                if old_key in data:
                    # 使用 .pop() 方法來獲取舊金鑰的值並同時從字典中移除它
                    data[new_key] = data.pop(old_key)

                # 將修改後的字典寫入新檔案
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    except FileNotFoundError:
        print(f"錯誤：找不到輸入檔案 '{input_path}'。請檢查路徑和檔名。")
        return

    print("\n" + "=" * 50)
    print("金鑰名稱更換完成！")
    print(f"新的檔案已儲存至: '{output_path}'")
    print("=" * 50)




def random_split_dataset(
        input_path: str = "credit_model_training_data.jsonl",
        train_output_path: str = "credit_train_split.jsonl",
        eval_output_path: str = "credit_eval_split.jsonl",
        split_ratio: float = 0.1  # 10% 作為驗證集
):
    """
    對 Credit Model 的訓練資料進行隨機分割。
    """
    print(f"正在從 '{input_path}' 讀取所有資料進行隨機分割...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            # 逐行讀取，避免一次性載入過大檔案到記憶體
            all_lines = f.readlines()

        if not all_lines:
            print("錯誤：輸入檔案為空，無法進行分割。")
            return

    except FileNotFoundError:
        print(f"錯誤：找不到輸入檔案 '{input_path}'。請檢查路徑。")
        return

    # 隨機打亂所有行的順序
    random.shuffle(all_lines)

    # 計算分割點
    split_index = int(len(all_lines) * (1.0 - split_ratio))

    train_lines = all_lines[:split_index]
    eval_lines = all_lines[split_index:]

    # 寫入新的訓練集檔案
    with open(train_output_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    # 寫入新的驗證集檔案
    with open(eval_output_path, 'w', encoding='utf-8') as f:
        f.writelines(eval_lines)

    print("\n" + "=" * 50)
    print("Credit Model 資料集隨機分割完成！")
    print(f"  總樣本數: {len(all_lines)}")
    print(f"  訓練集樣本數: {len(train_lines)} (已儲存至 '{train_output_path}')")
    print(f"  驗證集樣本數: {len(eval_lines)} (已儲存至 '{eval_output_path}')")
    print("=" * 50)


# --- 3. 主程式 ---
if __name__ == '__main__':
    #
    # # --- 參數設定 ---
    # INPUT_DATA_PATH = "/home/banlujie/data/bird_train/train/reward_train_split.jsonl"  # 使用您最終的、乾淨的配對資料集
    # DB_ROOT_PATH = "/home/banlujie/data/bird_train/train/train_databases"
    # OUTPUT_CREDIT_DATA_PATH = "/home/banlujie/data/bird_train/train/credit/credit_model_dataset.jsonl"
    # MAX_WORKERS = 4
    # TASK_TIMEOUT = 30
    # # --- 讀取輸入資料 ---
    # with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
    #     # 我們只需要 prompt 和 chosen，並且需要 db_id
    #     # 假設您的 reward_train_dataset.jsonl 中有 db_id
    #     tasks = [json.loads(line) for line in f]
    #
    # # --- 斷點續傳 ---
    # processed_prompts = set()
    # if os.path.exists(OUTPUT_CREDIT_DATA_PATH):
    #     print("發現進度檔案，正在讀取...")
    #     with open(OUTPUT_CREDIT_DATA_PATH, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             try:
    #                 processed_prompts.add(json.loads(line)['prompt'])
    #             except:
    #                 continue
    #     print(f"已讀取 {len(processed_prompts)} 個已處理的 prompt。")
    #
    # tasks_to_process = [task for task in tasks if task['prompt'] not in processed_prompts]
    # print(f"本次需處理 {len(tasks_to_process)} 個新 prompt。")
    # if not tasks_to_process:
    #     print("所有任務已完成。")
    #     exit()
    #
    # # --- 並行執行 ---
    # processed_count = 0
    #
    # with ProcessPool(max_workers=MAX_WORKERS) as pool, open(OUTPUT_CREDIT_DATA_PATH, 'a', encoding='utf-8') as f_out:
    #     print(f"正在將 {len(tasks_to_process)} 個任務提交到 Pebble 程序池...")
    #     # 提交任務到程序池
    #     future_map = {
    #         pool.schedule(
    #             function=process_clause_credit_task,
    #             args=(task, DB_ROOT_PATH),
    #             timeout=TASK_TIMEOUT
    #         ): task for task in tasks_to_process
    #     }
    #     progress_bar = tqdm(total=len(tasks_to_process), desc="計算子句信用")
    #
    #     for future in as_completed(future_map):
    #         original_task = future_map[future]
    #         try:
    #             # 獲取結果
    #             result = future.result()
    #             if result:
    #                 f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
    #         except TimeoutError:
    #             print(f"\n❌ 錯誤：任務因超時（超過 {TASK_TIMEOUT} 秒）而被中止。Prompt: {original_task['prompt'][:50]}...")
    #         except Exception as e:
    #             print(f"\n❌ 錯誤：任務執行失敗: {e}。Prompt: {original_task['prompt'][:50]}...")
    #
    #         # 更新進度條
    #         progress_bar.update(1)
    #     progress_bar.close()
    # print(f"✅ Credit Model 資料集建立完成，已儲存於: {OUTPUT_CREDIT_DATA_PATH}")
    # postprocess_for_model_training("/home/banlujie/data/bird_train/train/credit/credit_model_dataset.jsonl",
    #
    #                                "/home/banlujie/data/bird_train/train/credit/credit_model_dataset_training.jsonl")
    rename_key_in_jsonl(input_path="/home/banlujie/data/bird_train/train/credit/credit_model_dataset_training.jsonl",
                        output_path="/home/banlujie/data/bird_train/train/credit/credit_model_dataset_training_renamed.jsonl",)

    random_split_dataset(
        input_path="/home/banlujie/data/bird_train/train/credit/credit_model_dataset_training_renamed.jsonl",
        train_output_path="/home/banlujie/data/bird_train/train/credit/credit_train_split.jsonl",
        eval_output_path="/home/banlujie/data/bird_train/train/credit/credit_eval_split.jsonl",
        split_ratio=0.1
    )
