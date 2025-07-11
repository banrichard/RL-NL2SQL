import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os
import ray

from utils import execute_sql, serialize_schema, get_explicit_reward, extract_sql_from_output


@ray.remote(num_gpus=1)
class CandidateGeneratorActor:
    def __init__(self, sft_model_path: str, max_batch_size=256):
        self.max_batch_size = max_batch_size  # 每個 Actor 處理的最大批次大小
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path,
                                                       padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(sft_model_path).to(self.device)
        self.model.eval()
        self.gpu_id = ray.get_gpu_ids()[0]
        print(f"Actor 已在 GPU {self.gpu_id} 上成功初始化。")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_gpu_id(self):
        return self.gpu_id

    def generate_for_batch(self, prompts: list[str], num_candidates: int = 4):
        # 對收到的 prompts 批次進行編碼
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            generated_outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                num_return_sequences=num_candidates
            )

        # 解碼批次結果
        input_len = inputs['input_ids'].shape[1]
        decoded_sqls = self.tokenizer.batch_decode(generated_outputs[:, input_len:], skip_special_tokens=True)

        # 將扁平的結果列表重新組織成每個 prompt 對應一組候選答案
        # 例如，如果有 2 個 prompt，每個生成 4 個，decoded_sqls 長度為 8
        # 處理後會變成 [[sql1, sql2, sql3, sql4], [sql5, sql6, sql7, sql8]]
        results = []
        for i in range(len(prompts)):
            start_index = i * num_candidates
            end_index = (i + 1) * num_candidates
            prompt_candidates = [
                extract_sql_from_output(sql.strip()) for sql in decoded_sqls[start_index:end_index]
            ]
            results.append(list(set(filter(None, prompt_candidates))))

        return results


@ray.remote
def process_data_item(item, generator_actor, schema_map, db_root_path, prompt_template):
    db_id = item['db_id']
    chosen_sql = item['SQL']

    if db_id not in schema_map:
        return []  # --- 修改：回傳一個空列表

    # 1. 建立 Prompt
    schema_str = serialize_schema(schema_map[db_id])
    prompt_text = prompt_template.format(schema=schema_str, question=item['question'])

    # 2. 呼叫遠端的 Actor 進行模型推論
    candidates = ray.get(generator_actor.generate_for_prompt.remote(prompt_text, num_candidates=5))
    candidates = [extract_sql_from_output(raw_sql) for raw_sql in candidates]

    # 3. 驗證 SQL
    db_path = os.path.join(DB_ROOT_PATH, db_id, f"{db_id}.sqlite")
    chosen_result = execute_sql(db_path, chosen_sql)
    if type(chosen_result) is str and "Error" in chosen_result:
        return []

    # --- 修改開始：收集所有有效的 rejected 樣本 ---
    valid_pairs = []  # 建立一個空列表來存放結果
    score_chosen = 2.0
    for rejected_sql in candidates:
        if not rejected_sql or rejected_sql.lower() == chosen_sql.lower():
            continue

        score_rejected = get_explicit_reward(db_path, rejected_sql, chosen_sql, chosen_result)

        # 只要 rejected 的分數低於 chosen，就是一個有效的配對
        if score_rejected < score_chosen:
            # 計算分數差距
            margin = score_chosen - score_rejected

            reward_sample = {
                "prompt": prompt_text,
                "chosen": chosen_sql,
                "rejected": rejected_sql,
                "margin": margin  # 將 margin 加入樣本中
            }
            valid_pairs.append(reward_sample)

    return valid_pairs  # 回傳包含所有有效樣本的列表


# --- 使用範例 ---
if __name__ == '__main__':

    # 假設您想使用 4 張 GPU
    NUM_GPUS = 1
    MAX_BATCH_SIZE_PER_ACTOR = 2  # 每個 Actor 處理的最大批次大小
    NUM_CANDIDATES_PER_PROMPT = 5  # 每個 prompt 生成的候選數量
    # 初始化 Ray。如果您不指定資源，它會自動偵測所有可用的 CPU 和 GPU
    ray.init(num_gpus=NUM_GPUS)

    # 檢查 Ray 是否偵測到您期望的 GPU 數量
    available_gpus = ray.available_resources().get("GPU", 0)
    if available_gpus < NUM_GPUS:
        raise ValueError(f"錯誤：Ray 只偵測到 {available_gpus} 張 GPU，但您要求使用 {NUM_GPUS} 張。")
    print(f"Ray 已啟動，並偵測到 {available_gpus} 張 GPU。")

    SFT_MODEL_PATH = "/home/banlujie/RL-NL2SQL/checkpoint/phi4_mini_reasoning"
    TRAIN_PATH = "/home/banlujie/data/bird_train/train/train.json"
    TRAIN_TABLES_PATH = "/home/banlujie/data/bird_train/train/train_tables.json"
    OUTPUT_PATH = "/home/banlujie/data/bird_train/train/reward_ray_full.jsonl"
    DB_ROOT_PATH = "/home/banlujie/data/bird_train/train/train_databases"
    processed_tasks = set()

    if os.path.exists(OUTPUT_PATH):
        print(f"發現已存在的輸出檔案，正在讀取進度...")
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    # 我們用 prompt 作為唯一識別碼，因為它包含了 db_id 和 question
                    processed_prompt = json.loads(line)['prompt']
                    processed_tasks.add(processed_prompt)
                except (json.JSONDecodeError, KeyError):
                    continue  # 忽略格式錯誤的行
        print(f"已讀取 {len(processed_tasks)} 筆已完成的記錄。")
    # 載入資料
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(TRAIN_TABLES_PATH, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)

    schema_map = {db['db_id']: db for db in tables_data}
    prompt_template = """### Database Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL Query:\n"""
    db_root_path_ref = ray.put(DB_ROOT_PATH)

    # 1. 建立一個包含 4 個 Actor 的池 (Actor Pool)
    # Ray 會自動將這 4 個 Actor 分配到不同的可用 GPU 上
    print(f"正在建立 {NUM_GPUS} 個模型 Actor...")
    generator_actors = [
        CandidateGeneratorActor.remote(sft_model_path=SFT_MODEL_PATH, max_batch_size=MAX_BATCH_SIZE_PER_ACTOR) for _ in
        range(NUM_GPUS)]
    # 確認 Actor 是否已在不同 GPU 上啟動 (可選步驟)
    actor_gpus = ray.get([actor.get_gpu_id.remote() for actor in generator_actors])
    print(f"模型 Actor 已成功建立在 GPUs: {actor_gpus}")

    tasks_to_process = []
    for item in train_data:
        db_id = item['db_id']
        if db_id not in schema_map: continue
        schema_str = serialize_schema(schema_map[db_id])
        prompt_text = prompt_template.format(schema=schema_str, question=item['question'])

        # 如果這個 prompt 已經處理過，就跳過
        if prompt_text in processed_tasks:
            continue

        tasks_to_process.append({"prompt": prompt_text, "original_item": item})

    print(f"總任務數: {len(train_data)}, 已完成: {len(processed_tasks)}, 本次需處理: {len(tasks_to_process)}")
    # 2. 提交任務時，輪詢 (Round-Robin) 使用 Actor 池
    tasks_per_actor = [[] for _ in range(NUM_GPUS)]
    for i, task_meta in enumerate(tasks_to_process):
        tasks_per_actor[i % NUM_GPUS].append(task_meta)
    # 提交所有批次任務
    print("正在將任務分批並提交給各個 Actor...")
    result_futures = []
    for i in range(NUM_GPUS):
        actor = generator_actors[i]
        actor_tasks = tasks_per_actor[i]

        # 將分配給這個 Actor 的任務再切分成小批次
        for j in range(0, len(actor_tasks), MAX_BATCH_SIZE_PER_ACTOR):
            batch_tasks = actor_tasks[j: j + MAX_BATCH_SIZE_PER_ACTOR]
            batch_prompts = [task['prompt'] for task in batch_tasks]

            # 提交批次生成任務，並將原始資料 meta 一起傳遞
            future = actor.generate_for_batch.remote(batch_prompts, num_candidates=NUM_CANDIDATES_PER_PROMPT)
            result_futures.append((future, batch_tasks))

    # 3. 收集結果的邏輯不變
    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f_out:
        for future, batch_meta in tqdm(result_futures, desc="處理批次結果"):
            batch_results = ray.get(future)
            for i, candidates in enumerate(batch_results):
                item = batch_meta[i]['original_item']
                f_out.write(json.dumps({
                    "prompt": batch_meta[i]['prompt'], # 將 prompt 也寫入，作為唯一識別碼
                    "db_id": item['db_id'],
                    "chosen_sql": item['SQL'],
                    "candidate_sqls": candidates
                }) + "\n")
    ray.shutdown()
    print(f" 使用 {NUM_GPUS} 張 GPU 的並行化 Reward Dataset 生成完成！")
