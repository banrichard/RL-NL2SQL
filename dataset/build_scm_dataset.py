import os
import json
from tqdm import tqdm
from copy import deepcopy

# --- 1. 配置 ---
# 輸入路徑
ORIGINAL_DATA_FILE = "original_data.jsonl"
SCORED_RESPONSES_DIR = "scored_responses"
# 輸出路徑
OUTPUT_FILE = "scm_training_dataset.jsonl"


def create_dummy_files_if_not_exists():
    """為了讓腳本能直接運行，如果文件/資料夾不存在，則創建虛擬的樣本。"""
    if not os.path.exists(SCORED_RESPONSES_DIR):
        os.makedirs(SCORED_RESPONSES_DIR)

    # 虛擬的原始數據
    dummy_original_data = {
        "id": "request-1",
        "question": "Could you please list the names and total points of the players who are either the top scorer or the top rebounder?",
        "schema_columns": [
            "players.player_id", "players.player_name", "performance_stats.player_id",
            "performance_stats.ppg", "performance_stats.rpg"
        ]
    }
    # 虛擬的 Gemini 打標響應 (來自您提供的樣本)
    dummy_scored_data = {"step_wise_evaluation": [{"step": 1,
                                                   "description": "States the overall objective, identifying the need to find top players based on two stats and join tables to get their names.",
                                                   "focused_intent": ["PLANNING", "JOIN", "AGGREGATE"],
                                                   "convergence_score": 3, "efficiency_score": 9}, {"step": 2,
                                                                                                    "description": "Specifies the aggregation method, confirming that stats (points and rebounds) should be summed for each player.",
                                                                                                    "focused_intent": [
                                                                                                        "AGGREGATE",
                                                                                                        "GROUP_BY"],
                                                                                                    "convergence_score": 7,
                                                                                                    "efficiency_score": 10},
                                                  {"step": 3,
                                                   "description": "Makes the critical architectural decision to use a ranking mechanism, strongly implying the use of window functions like RANK() to handle the 'top' requirement for two separate criteria.",
                                                   "focused_intent": ["ORDER_BY", "PLANNING"], "convergence_score": 10,
                                                   "efficiency_score": 10}, {"step": 4,
                                                                             "description": "Defines the filtering logic for the final selection, specifying that players with a rank of 1 in EITHER category should be included.",
                                                                             "focused_intent": ["WHERE"],
                                                                             "convergence_score": 9,
                                                                             "efficiency_score": 9}, {"step": 5,
                                                                                                      "description": "Synthesizes the plan into a concrete query structure, detailing the CTE for ranking and the main query for final selection and filtering.",
                                                                                                      "focused_intent": [
                                                                                                          "SELECT",
                                                                                                          "FROM",
                                                                                                          "JOIN",
                                                                                                          "WHERE"],
                                                                                                      "convergence_score": 8,
                                                                                                      "efficiency_score": 7}],
                         "overall_justification": "The CoT demonstrates a high-quality, structured reasoning process...",
                         "final_parsed_intents": {"select": ["player_name", "total_points"], "from": ["players"],
                                                  "join": [{"table": "performance_stats",
                                                            "on": "p.player_id = ps.player_id"}],
                                                  "where": [{"condition": "points_rank = 1 OR rebounds_rank = 1"}],
                                                  "group_by": ["p.player_name"], "order_by": None,
                                                  "aggregate": [{"function": "SUM", "column": "ps.ppg"},
                                                                {"function": "SUM", "column": "ps.rpg"},
                                                                {"function": "RANK",
                                                                 "column": "OVER (ORDER BY SUM(ps.ppg) DESC)"},
                                                                {"function": "RANK",
                                                                 "column": "OVER (ORDER BY SUM(ps.rpg) DESC)"}]}}

    if not os.path.exists(ORIGINAL_DATA_FILE):
        print(f"創建虛擬原始數據文件: {ORIGINAL_DATA_FILE}")
        with open(ORIGINAL_DATA_FILE, 'w', encoding='utf-8') as f:
            f.write(json.dumps(dummy_original_data) + '\n')

    scored_file_path = os.path.join(SCORED_RESPONSES_DIR, f"{dummy_original_data['id']}.json")
    if not os.path.exists(scored_file_path):
        print(f"創建虛擬打標響應文件: {scored_file_path}")
        with open(scored_file_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_scored_data, f, indent=2)


def unroll_and_process_sample(original_item: dict, scored_item: dict) -> list:
    """
    將單個打標響應「展開」為多條訓練樣本的核心邏輯。
    """
    training_samples = []

    question = original_item["question"]
    schema_columns = original_item["schema_columns"]
    schema_str = " [SEP] ".join(schema_columns)

    step_wise_evaluation = scored_item["step_wise_evaluation"]
    final_parsed_intents = scored_item["final_parsed_intents"]

    cot_history = ""
    # 初始化一個空的意圖狀態，包含所有可能的鍵
    current_intent_state = {key: None for key in final_parsed_intents.keys()}

    # 遍歷 CoT 的每一步
    for step_data in step_wise_evaluation:
        # 1. 累積 CoT 歷史文本
        cot_history += step_data["description"] + " "

        # 2. 構建模型的輸入文本
        input_text = f"[CLS] {question} [SEP] {schema_str} [SEP] {cot_history.strip()} [SEP]"

        # 3. 根據 focused_intent，累積更新意圖標籤狀態
        for intent in step_data["focused_intent"]:
            intent_key = intent.lower()  # 轉換為小寫以匹配 final_parsed_intents 的鍵
            if intent_key in final_parsed_intents:
                current_intent_state[intent_key] = final_parsed_intents[intent_key]

        # 4. 提取品質分數標籤
        quality_scores = {
            "convergence": step_data["convergence_score"],
            "efficiency": step_data["efficiency_score"]
        }

        # 5. 組合成一條完整的訓練樣本
        # 使用 deepcopy 確保每個樣本的意圖狀態是獨立的快照
        sample = {
            "input_text": input_text,
            "labels": {
                "sql_intents": deepcopy(current_intent_state),
                "quality_scores": quality_scores
            },
            # 添加原始信息以便追溯和調試
            "source_id": original_item["id"],
            "step": step_data["step"]
        }
        training_samples.append(sample)

    return training_samples


def main():
    """主執行函數"""
    # 為了演示，自動創建所需的虛擬文件
    create_dummy_files_if_not_exists()

    # 1. 將原始數據加載到一個字典中，方便通過 ID 查找
    print(f"正在從 {ORIGINAL_DATA_FILE} 加載原始數據...")
    original_data_map = {}
    with open(ORIGINAL_DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            original_data_map[item['id']] = item

    # 2. 獲取所有已打標的響應文件名
    scored_files = os.listdir(SCORED_RESPONSES_DIR)
    print(f"在 {SCORED_RESPONSES_DIR} 中找到 {len(scored_files)} 個已打標文件。")

    # 3. 打開輸出文件，準備寫入
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        total_samples_generated = 0
        # 使用 tqdm 顯示處理進度
        for filename in tqdm(scored_files, desc="正在處理文件"):
            if not filename.endswith('.json'):
                continue

            file_id = filename.replace('.json', '')

            # 查找對應的原始數據
            if file_id not in original_data_map:
                print(f"警告：找不到 ID 為 '{file_id}' 的原始數據，跳過文件 {filename}")
                continue

            original_item = original_data_map[file_id]

            # 讀取打標後的 JSON 響應
            with open(os.path.join(SCORED_RESPONSES_DIR, filename), 'r', encoding='utf-8') as f_in:
                scored_item = json.load(f_in)

            # 4. 呼叫核心處理函數，將單個響應展開為多個樣本
            training_samples = unroll_and_process_sample(original_item, scored_item)

            # 5. 將生成的多個樣本逐行寫入輸出文件
            for sample in training_samples:
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                total_samples_generated += 1

    print(f"\n🎉 數據集構建完成！")
    print(f"從 {len(scored_files)} 個響應文件中，總共生成了 {total_samples_generated} 條訓練樣本。")
    print(f"結果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
