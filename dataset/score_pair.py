import random
import json
import os
import time
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from tqdm import tqdm
from utils import execute_sql_with_timeout, get_detailed_score


def process_entry(entry, db_root_path, processed_pairs):
    """
    處理單一完整的 entry (一個 prompt 及其所有候選 SQL)。
    這是優化後的核心函式，旨在減少重複計算。

    Args:
        entry (dict): 包含單一 prompt 完整資訊的字典。
        db_root_path (str): 資料庫根目錄的路徑。
        processed_pairs (set): 已經處理過的 (prompt, rejected_sql) 配對集合。

    Returns:
        list: 一個包含多個已生成配對字典的列表。如果 chosen_sql 失敗，則返回空列表。
    """
    pid = os.getpid()
    # print(f"工作進程 ID: {pid} 正在處理 db_id: {entry.get('db_id')}")

    db_id = entry.get('db_id')
    prompt = entry.get('prompt')
    chosen_sql = entry.get('chosen_sql')
    candidate_sqls = entry.get('candidate_sqls', [])

    if not all([db_id, prompt, chosen_sql]):
        return []

    db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")

    try:
        ground_truth_result = execute_sql_with_timeout(db_path, chosen_sql)
        if isinstance(ground_truth_result, str) and "Error" in ground_truth_result:
            return []
    except Exception:
        return []

    generated_pairs = []
    score_chosen = 2.0

    for rejected_sql in candidate_sqls:
        if not rejected_sql or rejected_sql.lower() == chosen_sql.lower():
            continue

        if (prompt, rejected_sql) in processed_pairs:
            continue

        rejected_details = get_detailed_score(
            db_path, rejected_sql, ground_truth_result)

        if rejected_details['score'] < score_chosen:
            margin = score_chosen - rejected_details['score']
            final_pair = {
                "prompt": prompt,
                "db_id": db_id,
                "chosen": chosen_sql,
                "rejected": rejected_sql,
                "margin": margin
            }
            generated_pairs.append(final_pair)

    return generated_pairs


def random_split_dataset(
    input_path: str = "reward_train_dataset_clean.jsonl",
    train_output_path: str = "reward_train_split.jsonl",
    eval_output_path: str = "reward_eval_split.jsonl",
    split_ratio: float = 0.1  # 10% 作為驗證集
):
    """
    對 reward dataset 進行隨機分割。
    """
    print(f"正在從 '{input_path}' 讀取所有資料進行隨機分割...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            # 逐行讀取，避免一次性載入過大檔案到記憶體
            all_lines = f.readlines()
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

    print("\n" + "="*50)
    print("資料集隨機分割完成！")
    print(f"  總樣本數: {len(all_lines)}")
    print(f"  訓練集樣本數: {len(train_lines)} (已儲存至 '{train_output_path}')")
    print(f"  驗證集樣本數: {len(eval_lines)} (已儲存至 '{eval_output_path}')")
    print("="*50)


def main():
    """
    主執行函式。
    """
    # --- 設定 ---
    # !! 請根據您的環境修改這些路徑 !!
    CANDIDATES_PATH = "/home/banlujie/data/synsql_5k/reward/reward_candidates_clean.jsonl"
    DB_ROOT_PATH = "/home/banlujie/data/synsql/databases"
    FINAL_PAIRED_PATH = "/home/banlujie/data/synsql_5k/reward/reward_train_dataset_new.jsonl"
    STUCK_LOG_PATH = "/home/banlujie/data/synsql_5k/reward/stuck_tasks_clean.log"
    MAX_WORKERS = 8
    # Pebble 的超時設定，這將是每個子進程任務的硬性限制
    TASK_TIMEOUT = 30

    # --- 強健的斷點續傳邏輯 ---
    processed_pairs = set()
    if os.path.exists(FINAL_PAIRED_PATH):
        print(f"發現已存在的輸出檔案 '{FINAL_PAIRED_PATH}'，正在讀取已完成的配對...")
        with open(FINAL_PAIRED_PATH, 'r', encoding='utf-8') as f_progress:
            for line in f_progress:
                try:
                    data = json.loads(line)
                    pair_id = (data['prompt'], data['rejected'])
                    processed_pairs.add(pair_id)
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"已讀取 {len(processed_pairs)} 筆已完成的配對。")

    # --- 讀取已知的卡住任務，建立黑名單 ---
    stuck_prompts = set()
    if os.path.exists(STUCK_LOG_PATH):
        print(f"發現卡住任務日誌 '{STUCK_LOG_PATH}'，正在讀取黑名單...")
        with open(STUCK_LOG_PATH, 'r', encoding='utf-8') as f_stuck_log:
            for line in f_stuck_log:
                try:
                    data = json.loads(line)
                    stuck_id = (data['db_id'], data['prompt'])
                    stuck_prompts.add(stuck_id)
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"已讀取 {len(stuck_prompts)} 筆已知會卡住的任務。")

    # --- 載入候選答案資料 ---
    print("正在載入候選答案資料...")
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f:
        candidate_data = [json.loads(line) for line in f]

    # --- 過濾掉已完成的和已知的卡住任務 ---
    print("正在過濾已完成及已知的卡住任務...")
    tasks_to_process = []
    for entry in tqdm(candidate_data, desc="檢查任務狀態"):
        db_id = entry.get('db_id')
        prompt = entry.get('prompt')

        if (db_id, prompt) in stuck_prompts:
            continue

        chosen_sql = entry.get('chosen_sql')
        if not all([prompt, chosen_sql]):
            continue

        needs_processing = any(
            (prompt, rej_sql) not in processed_pairs
            for rej_sql in entry.get('candidate_sqls', [])
            if rej_sql and rej_sql.lower() != chosen_sql.lower()
        )

        if needs_processing:
            tasks_to_process.append(entry)

    print(
        f"過濾完成。原共 {len(candidate_data)} 個 prompt，本次需要處理 {len(tasks_to_process)} 個。")

    if not tasks_to_process:
        print("所有任務皆已完成！")
        return

    # --- 使用 Pebble ProcessPool 處理任務 ---
    with open(FINAL_PAIRED_PATH, 'a', encoding='utf-8') as f_out, \
            open(STUCK_LOG_PATH, 'a', encoding='utf-8') as f_stuck, \
            ProcessPool(max_workers=MAX_WORKERS) as pool:

        # --- 提交任務到 Pebble 池 ---
        future_map = {}
        for entry in tasks_to_process:
            # 使用 Pebble 的 schedule 函式，並直接設定超時
            future = pool.schedule(
                process_entry,
                args=(entry, DB_ROOT_PATH, processed_pairs),
                timeout=TASK_TIMEOUT
            )
            future_map[future] = entry

        print(f"正在使用最多 {MAX_WORKERS} 個進程處理任務...")

        # --- 從 map 中獲取結果 ---
        for future, entry in tqdm(future_map.items(), total=len(tasks_to_process), desc="正在處理 Prompt"):
            try:
                list_of_pairs = future.result()  # Pebble 的 result() 會等待結果

                if list_of_pairs:
                    for pair in list_of_pairs:
                        f_out.write(json.dumps(
                            pair, ensure_ascii=False) + "\n")

            except TimeoutError:
                db_id = entry.get('db_id', 'N/A')
                prompt = entry.get('prompt', 'N/A')
                error_msg = f"警告：處理 db_id '{db_id}' 的任務因卡住而超時（超過 {TASK_TIMEOUT} 秒）。已跳過並記錄。"
                print(f"\n{error_msg}")

                f_stuck.write(json.dumps({
                    "error": f"Task timed out after {TASK_TIMEOUT} seconds.",
                    "db_id": db_id,
                    "prompt": prompt,
                    "chosen_sql": entry.get('chosen_sql', 'N/A')
                }, ensure_ascii=False) + "\n")
                f_stuck.flush()

            except Exception as e:
                db_id = entry.get('db_id', 'N/A')
                print(f"\n一個工作進程發生未知錯誤 (db_id: {db_id}): {e}")

    # 重新計算最終檔案中的總行數來給出準確計數
    with open(FINAL_PAIRED_PATH, 'r', encoding='utf-8') as f:
        final_count = sum(1 for _ in f)

    print(f"\n所有任務處理完成！最終資料集共有 {final_count} 筆配對。")
    print(f"資料已儲存於 '{FINAL_PAIRED_PATH}'。")
    print(f"卡住的任務已記錄於 '{STUCK_LOG_PATH}'。")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"總執行時間: {end_time - start_time:.2f} 秒")
    random_split_dataset(input_path="/home/banlujie/data/synsql_5k/reward/reward_train_dataset_new.jsonl", train_output_path="/home/banlujie/data/synsql_5k/reward/reward_train_split.jsonl", eval_output_path="/home/banlujie/data/synsql_5k/reward/reward_eval_split.jsonl", split_ratio=0.05)
