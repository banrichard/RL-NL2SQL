import json
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer


# --- 您的輔助函式 (不變) ---
from utils import serialize_schema, extract_sql_from_output


def main():
    # --- 參數設定 ---
    SFT_MODEL_PATH = "/home/banlujie/RL-NL2SQL/checkpoint/phi4_mini_reasoning"
    TRAIN_PATH = "/home/banlujie/data/bird_train/train/train_clean.json"
    TABLES_PATH = "/home/banlujie/data/bird_train/train/train_tables.json"
    CHAT_TEMPLATE_PATH = "/home/banlujie/RL-NL2SQL/templates/sql_generation.txt"
    OUTPUT_PATH = "/home/banlujie/data/bird_train/train/reward_candidates_clean.jsonl"

    # vLLM 的生成參數
    sampling_params = SamplingParams(
        n=5,
        temperature=0.7,
        top_p=0.95,
        max_tokens=4096,
        stop=["```","</answer>"]
    )

    # --- 1. 載入模型 (單次) ---
    # vLLM 會自動偵測並使用所有可見的 GPU
    # 如果您想限制 GPU 數量，請在執行前設定環境變數: export CUDA_VISIBLE_DEVICES=0,1
    print(f"正在從 '{SFT_MODEL_PATH}' 載入模型...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)


    # tensor_parallel_size 會自動設為可見的 GPU 數量
    llm = LLM(model=SFT_MODEL_PATH,tensor_parallel_size=torch.cuda.device_count())

    print("✅ 模型載入成功！")

    # --- 2. 準備所有 Prompts (一次性) ---
    print("正在準備所有 prompts...")
    with open(CHAT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        chat_template_string = f.read()
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(TABLES_PATH, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    schema_map = {db['db_id']: db for db in tables_data}

    tasks_to_process = []
    for item in train_data:
        db_id = item.get('db_id')
        if not db_id or db_id not in schema_map: continue
        schema_str = serialize_schema(schema_map[db_id])
        prompt_text = chat_template_string.format(
            db_engine="SQLite",
            db_details=schema_str,
            question=item['question']+ " " + item['evidence']
        )
        messages = [
            {"role": "user", "content": prompt_text}
        ]
        final_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        tasks_to_process.append({
            "prompt": final_prompt,
            "original_item": item
        })
    print(f"✅ 所有 {len(tasks_to_process)} 個 prompts 準備完成。")

    # (可選) 斷點續傳邏輯
    # ... 可以在此處加入斷點續傳邏輯，過濾 tasks_to_process ...

    # --- 3. 執行批次推論 (一次性呼叫) ---
    all_prompts = [task['prompt'] for task in tasks_to_process]

    print(f"\n正在對 {len(all_prompts)} 個 prompts 進行批次推論...")
    # 這是核心步驟，將所有 prompts 一次性傳入
    outputs = llm.generate(all_prompts, sampling_params)
    print("✅ 所有推論已完成！")

    # --- 4. 處理結果並寫入檔案 ---
    print("正在處理結果並寫入檔案...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        # outputs 的順序與輸入的 prompts 完全對應
        for i, output in enumerate(tqdm(outputs, desc="處理結果")):
            if i < 5:
                print("\n" + "#"*20 + f" 除錯輸出 for Prompt #{i+1} " + "#"*20)
                print(f"--- 原始 Prompt (結尾部分) ---")
                print(output.prompt[-300:]) # 印出 prompt 的最後 300 個字元
                print("\n--- vLLM 生成的原始候選答案 (Raw Completions) ---")
                for j, completion in enumerate(output.outputs):
                    print(f"  ➡️ 候選 {j+1}:")
                    # 使用 repr() 可以讓我們看到所有特殊字元，例如換行符 \n
                    print(f"    Raw Text: {repr(completion.text)}")
                    print(f"    Finish Reason: {completion.finish_reason}")
                print("#"*65)
            # --- 除錯日誌區塊結束 ---
            original_item = tasks_to_process[i]['original_item']

            candidates = [
                extract_sql_from_output(completion.text)
                for completion in output.outputs
            ]

            f_out.write(json.dumps({
                "prompt": output.prompt,
                "db_id": original_item['db_id'],
                "chosen_sql": original_item['SQL'],
                "candidate_sqls": list(set(filter(None, candidates)))
            }) + "\n")

    print(f"✅ 批次生成完成，已儲存於: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
