# generate_candidates_torchrun.py
import os

if "LOCAL_RANK" in os.environ:
    lr = int(os.environ["LOCAL_RANK"])
    # 若外部未指定 CVD，则按 LOCAL_RANK 指定单卡
    if "CUDA_VISIBLE_DEVICES" not in os.environ or not os.environ["CUDA_VISIBLE_DEVICES"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(lr)
    else:
        # 若外部已经设定了多张卡，可按索引缩减为当前进程的那一张
        vis = [x.strip() for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip() != ""]
        if 0 <= lr < len(vis):
            os.environ["CUDA_VISIBLE_DEVICES"] = vis[lr]
        else:
            # 兜底：至少保证有一张卡可见
            os.environ["CUDA_VISIBLE_DEVICES"] = vis[0]
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

import numpy as np
from datasets import load_dataset
from utils import serialize_schema, extract_sql_from_output

from contextlib import contextmanager

@contextmanager
def _isolated_env_for_vllm():
    """在创建 LLM 前临时移除 torchrun 的分布式环境变量，避免 vLLM 误用。"""
    keys = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"]
    saved = {k: os.environ[k] for k in keys if k in os.environ}
    try:
        for k in keys:
            os.environ.pop(k, None)
        yield
    finally:
        os.environ.update(saved)

def setup_distributed():
    """初始化分散式環境"""
    # torchrun 會自動設定環境變數
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 雖然推論不一定需要初始化 process_group，但這是標準做法
    # dist.init_process_group(backend="nccl")

    # 將目前程序鎖定到對應的 GPU
    torch.cuda.set_device(local_rank)

    return local_rank, world_size


def generate_and_write(
        local_rank: int,
        tasks: list,
        sft_model_path: str,
        max_batch_size: int,
        num_candidates: int,
        temp_output_path: str
):
    """
    這個函式包含了模型載入和生成的核心邏輯。
    """
    with _isolated_env_for_vllm():
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=sft_model_path,
            tensor_parallel_size=1,          # 每进程1卡
            gpu_memory_utilization=0.3,
            trust_remote_code=True,
            # enforce_eager=True,            # 若初始化仍卡，先打开这一行定位问题
        )
        sampling = SamplingParams(
            max_tokens=8192,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            n=num_candidates,
        )
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, padding_side='left')

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    llm = LLM(
        model=sft_model_path,
        tensor_parallel_size=1,              # 每个进程1卡
        gpu_memory_utilization=0.4,
        trust_remote_code=True               # 若模型需要自定义代码
    )
    sampling = SamplingParams(
        max_tokens=8192,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        n=num_candidates,                    # 每个 prompt 生成 n 个候选
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"程序 RANK {local_rank} (PID: {os.getpid()}) 已在 GPU {local_rank} 上成功初始化模型。")
    processed_prompts = set()
    if os.path.exists(temp_output_path):
        print(f"RANK {local_rank}: 發現進度檔案，正在讀取...")
        with open(temp_output_path, 'r', encoding='utf-8') as f_progress:
            for line in f_progress:
                try:
                    processed_prompts.add(json.loads(line)['prompt'])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"RANK {local_rank}: 已讀取 {len(processed_prompts)} 筆已完成記錄。")

    # --- 斷點續傳修改 (2/2): 過濾已完成的任務 ---
    tasks_to_process = [task for task in tasks if task['prompt'] not in processed_prompts]
    print(f"RANK {local_rank}: 本次需處理 {len(tasks_to_process)} 個新任務。")

    if not tasks_to_process:
        return  # 如果這個分片的所有任務都已完成，直接退出

    with open(temp_output_path, 'a', encoding='utf-8') as f_out:
        # 只有 rank 0 的程序顯示總進度條
        if local_rank == 0:
            pbar = tqdm(total=len(tasks) * int(os.environ["WORLD_SIZE"]), desc="總體進度")
        pbar_desc = f"GPU {local_rank} 進度"
        for i in tqdm(range(0, len(tasks_to_process), max_batch_size), desc=pbar_desc, position=local_rank):
            batch_tasks = tasks_to_process[i: i + max_batch_size]
            batch_prompts = [task['prompt'] for task in batch_tasks]

            request_outputs = llm.generate(batch_prompts, sampling_params=sampling)

            results = []
            for req_out in request_outputs:
                prompt_candidates = [
                    extract_sql_from_output(out.text.strip())
                    for out in req_out.outputs
                ]
                results.append(list(set(filter(None, prompt_candidates))))

            for j, candidates in enumerate(results):
                item_meta = batch_tasks[j]
                item = item_meta['original_item']
                f_out.write(json.dumps({
                    "prompt": item_meta['prompt'],
                    "db_id": item['db_id'],
                    "chosen_sql": item['SQL'] if 'SQL' in item else item.get('sql', ''),
                    "candidate_sqls": candidates
                }) + "\n")

            if local_rank == 0:
                pbar.update(len(batch_tasks) * int(os.environ["WORLD_SIZE"]))

        if local_rank == 0:
            pbar.close()


def main():
    # 1. 初始化分散式環境並獲取身份
    local_rank, world_size = setup_distributed()

    # 參數設定
    SFT_MODEL_PATH = "seeklhy/OmniSQL-7B"
    TRAIN_PATH = "/home/banlujie/data/synsql_5k/train.jsonl"
    TABLES_PATH = "/home/banlujie/data/bird_train/train/train_tables.json"
    OUTPUT_PATH = "/home/banlujie/data/synsql_5k/reward/reward_candidates_clean.jsonl"
    BUILD_BIRD = False
    MAX_BATCH_SIZE_PER_GPU = 4
    NUM_CANDIDATES_PER_PROMPT = 5

    # 只有主程序 (rank 0) 執行一次資料載入和準備
    if local_rank == 0:
        print("主程序 (RANK 0)：正在載入與準備資料...")
        if BUILD_BIRD:
            with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
                train_data = json.load(f)

            with open(TABLES_PATH, 'r', encoding='utf-8') as f:
                tables_data = json.load(f)
            schema_map = {db['db_id']: db for db in tables_data}
            prompt_template = """### Database Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL Query:\n"""

            all_tasks = []
            for item in train_data:
                db_id = item['db_id']
                if db_id not in schema_map: continue
                schema_str = serialize_schema(schema_map[db_id])
                prompt_text = prompt_template.format(schema=schema_str, question=item['question'])
                all_tasks.append({"prompt": prompt_text, "original_item": item})
        else:
            train_data = []
            with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        train_data.append(json.loads(line))
                        # 在这里处理每一行的数据
                    except json.JSONDecodeError:
                        print(f"Could not decode line: {line}")
            all_tasks = []
            for item in train_data:
                all_tasks.append({"prompt": item['prompt'], "original_item": item})
        print(f"資料準備完成，共 {len(all_tasks)} 個任務。")
    else:
        all_tasks = None

    # 使用廣播 (broadcast) 將 all_tasks 從 rank 0 發送給所有其他程序
    # 這可以避免每個程序都重複讀取檔案
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        tasks_to_broadcast = [all_tasks]
        dist.broadcast_object_list(tasks_to_broadcast, src=0)
        all_tasks = tasks_to_broadcast[0]

    # 2. 每個程序根據自己的 rank 計算要處理的資料分片
    total_tasks = len(all_tasks)
    tasks_per_proc = total_tasks // world_size
    start_index = local_rank * tasks_per_proc
    end_index = (local_rank + 1) * tasks_per_proc if local_rank < world_size - 1 else total_tasks
    my_tasks = all_tasks[start_index:end_index]

    print(f"RANK {local_rank}: 已分配到 {len(my_tasks)} 個任務 (從索引 {start_index} 到 {end_index})。")

    # 3. 執行生成任務
    temp_output_path = f"{OUTPUT_PATH}.part_{local_rank}"
    generate_and_write(local_rank, my_tasks, SFT_MODEL_PATH, MAX_BATCH_SIZE_PER_GPU, NUM_CANDIDATES_PER_PROMPT,
                       temp_output_path)

    # 4. 等待所有程序都完成
    if world_size > 1:
        dist.barrier()

    # 5. 只有主程序 (rank 0) 負責合併檔案
    if local_rank == 0:
        print("所有工作程序已完成。正在合併暫存檔案...")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
            for i in range(world_size):
                temp_file = f"{OUTPUT_PATH}.part_{i}"
                if os.path.exists(temp_file):
                    with open(temp_file, 'r', encoding='utf-8') as f_in:
                        for line in f_in:
                            f_out.write(line)
                    os.remove(temp_file)
        print(f" 使用 {world_size} 張 GPU 的並行化生成完成！最終檔案已儲存於: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
