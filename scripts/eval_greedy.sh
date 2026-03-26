#!/bin/bash

# 贪心策略评估脚本

echo "=== 开始评估模型（贪心策略）==="

MODEL_PATH="${1:-microsoft/Phi-4-mini-reasoning}"
BACKEND="${BACKEND:-vllm}"
SGLANG_MODEL="${SGLANG_MODEL:-$MODEL_PATH}"
echo "使用模型: $MODEL_PATH"
echo "推理后端: $BACKEND"

if [ "$BACKEND" = "sglang" ]; then
    python3 infer_sglang.py \
        --model "$SGLANG_MODEL" \
        --dev_path "/data/banlujie/data/bird_dev/dev.json" \
        --tables_path "/data/banlujie/data/bird_dev/dev_tables.json" \
        --db_root_path "/data/banlujie/data/bird_dev/dev_databases" \
        --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
        --system_prompt_path "/data/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
        --output_name "/data/banlujie/RL-NL2SQL/eval_res/eval_results_greedy" \
        --strategy "greedy" \
        --temperature 0.0 \
        --max_tokens 8192
elif [ "$BACKEND" = "transformers" ]; then
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 infer_transformers.py \
        --model_path "$MODEL_PATH" \
        --dev_path "/data/banlujie/data/bird_dev/dev.json" \
        --tables_path "/data/banlujie/data/bird_dev/dev_tables.json" \
        --db_root_path "/data/banlujie/data/bird_dev/dev_databases" \
        --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
        --system_prompt_path "/data/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
        --output_name "/data/banlujie/RL-NL2SQL/eval_res/eval_results_greedy" \
        --strategy "greedy" \
        --temperature 0.0 \
        --max_tokens 2048 \
        --batch_size 32 \
        --torch_dtype bfloat16
else
    CUDA_VISIBLE_DEVICES=5,7 python3 infer.py \
        --model_path "$MODEL_PATH" \
        --dev_path "/data/banlujie/data/bird_dev/dev.json" \
        --tables_path "/data/banlujie/data/bird_dev/dev_tables.json" \
        --db_root_path "/data/banlujie/data/bird_dev/dev_databases" \
        --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
        --system_prompt_path "/data/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
        --output_name "/data/banlujie/RL-NL2SQL/eval_res/eval_results_greedy" \
        --strategy "greedy" \
        --temperature 0.0 \
        --max_tokens 8192 \
        --timeout 10
fi

echo "=== 贪心策略评估完成 ==="
