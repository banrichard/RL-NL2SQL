#!/bin/bash


echo "=== 开始评估 Qwen2.5-coder-instruct-7b 模型（Hugging Face 原始版本）==="

# 使用 Hugging Face 模型 ID，vLLM 会自动下载
MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"

echo "使用模型: $MODEL_PATH"
echo "注意：首次运行时会自动从 Hugging Face 下载模型，可能需要较长时间"

python eval_vllm.py \
    --model_path "$MODEL_PATH" \
    --dev_path "/home/banlujie/data/bird_dev/dev.json" \
    --tables_path "/home/banlujie/data/bird_dev/dev_tables.json" \
    --db_root_path "/home/banlujie/data/bird_dev/dev_databases" \
    --template_path "/home/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
    --system_prompt_path "/home/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
    --output_path "/home/banlujie/RL-NL2SQL/eval_results_qwen2.5_7b_instruct.json" \
    --temperature 0.1 \
    --max_tokens 16384 \
    --timeout 10

echo "=== 评估完成 ==="
echo "结果保存在: /home/banlujie/RL-NL2SQL/eval_results_qwen2.5_7b_instruct.json"
