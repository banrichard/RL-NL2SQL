import json
import os
from tqdm import tqdm
from utils import  get_detailed_score, execute_sql_with_timeout
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError


def process_and_score_item(candidate_entry, db_root_path):
    """
    處理單一 prompt 及其所有候選答案的函式。
    """
    db_id = candidate_entry.get('db_id')
    prompt = candidate_entry.get('prompt')
    chosen_sql = candidate_entry.get('chosen_sql')
    candidate_sqls = candidate_entry.get('candidate_sqls', [])

    if not all([db_id, prompt, chosen_sql]):
        return None

    db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")

    try:
        ground_truth_result = execute_sql_with_timeout(db_path, chosen_sql)
        if isinstance(ground_truth_result, str) and "Error" in ground_truth_result:
            raise ValueError(f"Ground truth SQL failed: {ground_truth_result}")
    except Exception as e:
        print(f"警告：標準答案執行失敗，db_id: {db_id}, error: {e}")
        return None

    score_chosen = 2.0
    scored_chosen = {"sql": chosen_sql, "score": score_chosen, "is_correct": True, "error": None}

    scored_candidates = []
    for rejected_sql in candidate_sqls:
        if rejected_sql.lower() == chosen_sql.lower():
            continue

        rejected_details = get_detailed_score(db_path, rejected_sql, ground_truth_result)
        margin = score_chosen - rejected_details['score']
        rejected_details['margin_vs_chosen'] = margin
        scored_candidates.append(rejected_details)

    return {
        "prompt": prompt,
        "chosen_sample": scored_chosen,
        "rejected_samples": scored_candidates
    }


if __name__ == "__main__":
    CANDIDATES_PATH = "/home/banlujie/data/bird_train/train/reward_ray_full.jsonl"
    DB_ROOT_PATH = "/home/banlujie/data/bird_train/train/train_databases"
    FINAL_PAIRED_PATH = "/home/banlujie/data/bird_train/train/reward_train_dataset.jsonl"  # 這是最終給 OpenRLHF 的檔案

    # --- 強健的斷點續傳邏輯 ---
    processed_pairs = set()
    if os.path.exists(FINAL_PAIRED_PATH):
        print(f"發現已存在的輸出檔案 '{FINAL_PAIRED_PATH}'，正在讀取已完成的配對...")
        with open(FINAL_PAIRED_PATH, 'r', encoding='utf-8') as f_progress:
            for line in f_progress:
                try:
                    # 讀取最終檔案的簡單格式
                    data = json.loads(line)
                    # 建立一個由 prompt 和 rejected SQL 組成的唯一標識
                    pair_id = (data['prompt'], data['rejected'])
                    processed_pairs.add(pair_id)
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"已讀取 {len(processed_pairs)} 筆已完成的配對。")

    # 載入候選答案資料
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f:
        candidate_data = [json.loads(line) for line in f]

    tasks_to_process = []
    print("正在計算需要處理的剩餘任務...")
    for entry in candidate_data:
        prompt = entry.get('prompt')
        chosen_sql = entry.get('chosen_sql')

        if not prompt or not chosen_sql:
            continue

        for candidate_sql in entry.get('candidate_sqls', []):
            if not candidate_sql or candidate_sql.lower() == chosen_sql.lower():
                continue
            # 建立配對標識
            current_pair_id = (prompt, candidate_sql)
            # 如果這個配對是全新的，才將其加入待辦列表
            if current_pair_id not in processed_pairs:
                tasks_to_process.append({
                    "db_id": entry['db_id'],
                    "prompt": prompt,
                    "chosen_sql": chosen_sql,
                    "rejected_sql": candidate_sql  # 只儲存需要處理的這一個 rejected
                })

    print(f"計算完成。本次需要處理 {len(tasks_to_process)} 個新的配對。")

    if not tasks_to_process:
        print("所有任務皆已完成！")
        exit()

    # 以附加模式 ('a') 開啟檔案
    with open(FINAL_PAIRED_PATH, 'a', encoding='utf-8') as f_out:
        for task in tqdm(tasks_to_process, desc="正在處理剩餘任務"):
            db_id = task['db_id']
            prompt = task['prompt']
            chosen_sql = task['chosen_sql']
            rejected_sql = task['rejected_sql']

            db_path = os.path.join(DB_ROOT_PATH, db_id, f"{db_id}.sqlite")

            # 獲取標準答案的執行結果（這一步會有重複計算，但對於單個 prompt 來說開銷可接受）
            try:
                ground_truth_result = execute_sql_with_timeout(db_path, chosen_sql)
                if isinstance(ground_truth_result, str) and "Error" in ground_truth_result:
                    continue  # 如果標準答案有問題，跳過基於它的所有配對
            except Exception:
                continue

            # 為 rejected SQL 打分
            score_chosen = 2.0
            rejected_details = get_detailed_score(db_path, rejected_sql, ground_truth_result)

            # 建立最終的配對並寫入
            if rejected_details['score'] < score_chosen:
                margin = score_chosen - rejected_details['score']
                final_pair = {
                    "prompt": prompt,
                    "chosen": chosen_sql,
                    "rejected": rejected_sql,
                    "margin": margin
                }
                f_out.write(json.dumps(final_pair) + "\n")

    print(f" 所有任務處理完成！最終資料集大小為 {len(processed_pairs)} 筆，已儲存於 '{FINAL_PAIRED_PATH}'。")