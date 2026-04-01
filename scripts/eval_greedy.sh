#!/bin/bash

# 贪心策略评估脚本
# 用法:
#   bash scripts/eval_greedy.sh [MODEL_PATH] [DATASET]
#   DATASET 可选: bird_dev (默认), bird_train

echo "=== 开始评估模型（贪心策略）==="

MODEL_PATH="${1:-/data/banlujie/ckpt/Qwen3.5-9B}"
DATASET="${2:-bird_train}"
BACKEND="${BACKEND:-sglang}"
SGLANG_MODEL="${SGLANG_MODEL:-$MODEL_PATH}"

# --- SGLang GPU / TP 设置（仿照 eval_voting.sh） ---
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-1,3}"
if [ -z "${SGLANG_TP_SIZE:-}" ]; then
    _SGLANG_GPU_LIST="${SGLANG_CUDA_VISIBLE_DEVICES// /}"
    IFS=',' read -r -a _SGLANG_GPU_ARRAY <<< "$_SGLANG_GPU_LIST"
    SGLANG_TP_SIZE="${#_SGLANG_GPU_ARRAY[@]}"
    if [ -z "$SGLANG_TP_SIZE" ] || [ "$SGLANG_TP_SIZE" -lt 1 ]; then
        SGLANG_TP_SIZE=1
    fi
fi
_SGLANG_GPU_LIST="${SGLANG_CUDA_VISIBLE_DEVICES// /}"
IFS=',' read -r -a _SGLANG_GPU_ARRAY <<< "$_SGLANG_GPU_LIST"
SGLANG_VISIBLE_GPU_COUNT="${#_SGLANG_GPU_ARRAY[@]}"
if [ "$SGLANG_TP_SIZE" -ne "$SGLANG_VISIBLE_GPU_COUNT" ]; then
    echo "警告: SGLANG_TP_SIZE($SGLANG_TP_SIZE) 与可见GPU数量($SGLANG_VISIBLE_GPU_COUNT)不一致，已自动修正"
    SGLANG_TP_SIZE="$SGLANG_VISIBLE_GPU_COUNT"
fi

# --- 数据集路径设置 ---
MODEL_NAME="$(basename "$MODEL_PATH")"

if [ "$DATASET" = "bird_train" ]; then
    DEV_PATH="/data/banlujie/data/bird_train/train/train.json"
    TABLES_PATH="/data/banlujie/data/bird_train/train/train_tables.json"
    DB_ROOT_PATH="/data/banlujie/data/bird_train/train/train_databases"
    OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_greedy_bird_train"
    GOLD_FILE="$DEV_PATH"
    DB_EVAL_PATH="$DB_ROOT_PATH"
else
    DEV_PATH="/data/banlujie/data/bird_dev/dev.json"
    TABLES_PATH="/data/banlujie/data/bird_dev/dev_tables.json"
    DB_ROOT_PATH="/data/banlujie/data/bird_dev/dev_databases"
    OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_greedy"
    GOLD_FILE="$DEV_PATH"
    DB_EVAL_PATH="$DB_ROOT_PATH"
fi

PRED_FILE="${OUTPUT_PREFIX}_${MODEL_NAME}.json"

echo "使用模型: $MODEL_PATH"
echo "数据集: $DATASET"
echo "推理后端: $BACKEND"
echo "SGLang 可见GPU: $SGLANG_CUDA_VISIBLE_DEVICES"
echo "SGLang Tensor Parallel: $SGLANG_TP_SIZE"
echo "推理输出文件: $PRED_FILE"

if [ "$BACKEND" = "sglang" ]; then
    source /data/banlujie/sglang/bin/activate
    SGLANG_DISABLE_CUDNN_CHECK=1 CUDA_VISIBLE_DEVICES="$SGLANG_CUDA_VISIBLE_DEVICES" python3 infer_sglang.py \
        --model "$SGLANG_MODEL" \
        --dp-size "$SGLANG_TP_SIZE" \
        --dataset "$DATASET" \
        --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation.txt" \
        --output_name "$OUTPUT_PREFIX" \
        --strategy "greedy" \
        --temperature 0.0 \
        --max_tokens 4096
elif [ "$BACKEND" = "transformers" ]; then
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 infer_transformers.py \
        --model_path "$MODEL_PATH" \
        --dev_path "$DEV_PATH" \
        --tables_path "$TABLES_PATH" \
        --db_root_path "$DB_ROOT_PATH" \
        --template_path "/home/banlujie/RL-NL2SQL/templates/sql_generation.txt" \
        --system_prompt_path "/data/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
        --output_name "$OUTPUT_PREFIX" \
        --strategy "greedy" \
        --temperature 0.0 \
        --max_tokens 2048 \
        --batch_size 32 \
        --torch_dtype bfloat16
else
    CUDA_VISIBLE_DEVICES=5,7 python3 infer.py \
        --model_path "$MODEL_PATH" \
        --dev_path "$DEV_PATH" \
        --tables_path "$TABLES_PATH" \
        --db_root_path "$DB_ROOT_PATH" \
        --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation.txt" \
        --system_prompt_path "/data/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
        --output_name "$OUTPUT_PREFIX" \
        --strategy "greedy" \
        --temperature 0.0 \
        --max_tokens 8192 \
        --timeout 10
fi

# 运行 eval_bird.py 评估
python3 eval_bird.py \
    --pred "$PRED_FILE" \
    --gold "$GOLD_FILE" \
    --db_path "$DB_EVAL_PATH" \
    --mode "greedy_search"

echo "=== 贪心策略评估完成 ==="
echo "数据集: $DATASET"
echo "结果来源文件: $PRED_FILE"
