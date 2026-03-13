#!/bin/bash

# 贪心策略评估脚本

echo "=== 开始评估模型（贪心策略）==="

MODEL_PATH="${1:-microsoft/Phi-4-mini-reasoning}"
echo "使用模型: $MODEL_PATH"

CUDA_VISIBLE_DEVICES=0,1 python3 infer.py \
    --model_path "$MODEL_PATH" \
    --dev_path "/home/banlujie/data/bird_dev/dev.json" \
    --tables_path "/home/banlujie/data/bird_dev/dev_tables.json" \
    --db_root_path "/home/banlujie/data/bird_dev/dev_databases" \
    --template_path "/home/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
    --system_prompt_path "/home/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
    --output_name "/home/banlujie/RL-NL2SQL/eval_res/eval_results_greedy" \
    --strategy "greedy" \
    --temperature 0.0 \
    --max_tokens 8192 \
    --timeout 10

echo "=== 贪心策略评估完成 ==="
