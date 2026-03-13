#!/bin/bash

# 投票策略评估脚本
# seeklhy/OmniSQL-7B
echo "=== 开始评估模型（投票策略）==="

MODEL_PATH="${1:-/home/banlujie/RL-NL2SQL/ckpt/omnisql-7b-ppo}"
NUM_CANDIDATES="${2:-8}"
OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_voting"
MODEL_NAME="$(basename "$MODEL_PATH")"
PRED_FILE="${OUTPUT_PREFIX}_${MODEL_NAME}.json"
echo "使用模型: $MODEL_PATH"
echo "候选数量: $NUM_CANDIDATES"
echo "推理输出文件: $PRED_FILE"

CUDA_VISIBLE_DEVICES=1 python3 infer.py \
    --model_path "$MODEL_PATH" \
    --dev_path "/data/banlujie/data/bird_dev/dev_omni.json" \
    --tables_path "/data/banlujie/data/bird_dev/dev_tables.json" \
    --db_root_path "/data/banlujie/data/bird_dev/dev_databases" \
    --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
    --system_prompt_path "/data/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
    --output_name "$OUTPUT_PREFIX" \
    --strategy "voting" \
    --num_candidates "$NUM_CANDIDATES" \
    --temperature 0.8 \
    --max_tokens 8192 \
    
python3 eval_bird.py \
    --pred "$PRED_FILE" \
    --mode "major_voting"\


echo "=== 投票策略评估完成 ==="
echo "结果来源文件: $PRED_FILE"
