#!/bin/bash
# Spider 投票策略评估脚本
#
# 用法:
#   bash scripts/eval_spider_voting.sh [MODEL_PATH] [NUM_CANDIDATES] [DATASET]
#
# 环境变量覆盖（与 eval_voting.sh 对齐）:
#   SGLANG_CUDA_VISIBLE_DEVICES, SGLANG_TP_SIZE, SGLANG_GPU_UTILIZATION
#   SGLANG_MAX_TOKENS, SGLANG_BATCH_SIZE, SGLANG_MODEL
#   EVAL_TIMEOUT, EVAL_NUM_CPUS, EVAL_DISABLE_POST_PROCESS

set -euo pipefail

echo "=== 开始 Spider 评估（投票策略）==="

MODEL_PATH="${1:-/data/banlujie/ckpt/Qwen3.5-9B}"
NUM_CANDIDATES="${2:-8}"
DATASET="${3:-spider_test}"
SGLANG_MODEL="${SGLANG_MODEL:-$MODEL_PATH}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30}"
EVAL_NUM_CPUS="${EVAL_NUM_CPUS:-20}"
EVAL_DISABLE_POST_PROCESS="${EVAL_DISABLE_POST_PROCESS:-1}"
SGLANG_GPU_UTILIZATION="${SGLANG_GPU_UTILIZATION:-0.6}"
SGLANG_MAX_TOKENS="${SGLANG_MAX_TOKENS:-4096}"
SGLANG_BATCH_SIZE="${SGLANG_BATCH_SIZE:-}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-}"
SGLANG_QUANTIZATION_PARAM_PATH="${SGLANG_QUANTIZATION_PARAM_PATH:-}"
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-1}"
PARALLEL_MODE="${PARALLEL_MODE:-tp}"   # tp | dp | dp+tp

# --- GPU 数量 ---
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-6,7}"
IFS=',' read -r -a _GPU_ARRAY <<< "$SGLANG_CUDA_VISIBLE_DEVICES"
_GPU_COUNT="${#_GPU_ARRAY[@]}"
[ "$_GPU_COUNT" -lt 1 ] && _GPU_COUNT=1

# --- 根据并行模式确定 TP / DP ---
case "$PARALLEL_MODE" in
    tp)
        SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-$_GPU_COUNT}"
        SGLANG_DP_SIZE="${SGLANG_DP_SIZE:-1}"
        ;;
    dp)
        SGLANG_DP_SIZE="${SGLANG_DP_SIZE:-$_GPU_COUNT}"
        SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"
        ;;
    dp+tp)
        if [ -z "${SGLANG_DP_SIZE:-}" ] || [ -z "${SGLANG_TP_SIZE:-}" ]; then
            echo "错误: PARALLEL_MODE=dp+tp 时必须同时设置 SGLANG_DP_SIZE 和 SGLANG_TP_SIZE"
            exit 1
        fi
        _REQUIRED=$(( SGLANG_DP_SIZE * SGLANG_TP_SIZE ))
        if [ "$_REQUIRED" -ne "$_GPU_COUNT" ]; then
            echo "错误: SGLANG_DP_SIZE(${SGLANG_DP_SIZE}) × SGLANG_TP_SIZE(${SGLANG_TP_SIZE}) = ${_REQUIRED}，但可见 GPU 数量为 ${_GPU_COUNT}"
            exit 1
        fi
        ;;
    *)
        echo "错误: 未知 PARALLEL_MODE='$PARALLEL_MODE'，可选值: tp | dp | dp+tp"
        exit 1
        ;;
esac

# --- 数据集路径 ---
MODEL_NAME="$(basename "$MODEL_PATH")"
if [ "$DATASET" = "spider_test" ]; then
    DEV_PATH="/data/banlujie/data/spider/test.json"
    DB_ROOT_PATH="/data/banlujie/data/spider/test_database"
    GOLD_FILE="$DEV_PATH"
    OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_spider_voting_test"
else
    DEV_PATH="/data/banlujie/data/spider/dev.json"
    DB_ROOT_PATH="/data/banlujie/data/spider/database"
    GOLD_FILE="$DEV_PATH"
    OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_spider_voting"
fi

PRED_FILE="${OUTPUT_PREFIX}_${MODEL_NAME}.json"

echo "模型: $MODEL_PATH"
echo "SGLang 推理模型: $SGLANG_MODEL"
if [ "$SGLANG_MODEL" != "$MODEL_PATH" ]; then
    echo "警告: SGLang 推理模型与评测模型路径不一致，结果可能与预期不符"
fi
echo "候选数量: $NUM_CANDIDATES"
echo "数据集: $DATASET"
echo "并行模式: $PARALLEL_MODE  (TP=${SGLANG_TP_SIZE}, DP=${SGLANG_DP_SIZE})"
echo "关闭 eval 后处理: $EVAL_DISABLE_POST_PROCESS"
echo "SGLang 可见GPU: $SGLANG_CUDA_VISIBLE_DEVICES"
echo "SGLang GPU 利用率: $SGLANG_GPU_UTILIZATION"
echo "SGLang Max Tokens: $SGLANG_MAX_TOKENS"
echo "SGLang Batch Size: ${SGLANG_BATCH_SIZE:-default}"
echo "SGLang KV Cache DType: ${SGLANG_KV_CACHE_DTYPE:-auto(default)}"
echo "SGLang Quant Param Path: ${SGLANG_QUANTIZATION_PARAM_PATH:-none}"
echo "SGLang TP 显存均衡检查: $SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"
echo "推理输出文件: $PRED_FILE"
echo ""

SGLANG_EXTRA_ARGS=()
[ -n "$SGLANG_KV_CACHE_DTYPE" ] && SGLANG_EXTRA_ARGS+=(--kv-cache-dtype "$SGLANG_KV_CACHE_DTYPE")
[ -n "$SGLANG_QUANTIZATION_PARAM_PATH" ] && SGLANG_EXTRA_ARGS+=(--quantization-param-path "$SGLANG_QUANTIZATION_PARAM_PATH")
[ -n "$SGLANG_BATCH_SIZE" ] && SGLANG_EXTRA_ARGS+=(--batch_size "$SGLANG_BATCH_SIZE")

EVAL_EXTRA_ARGS=()
[ "$EVAL_DISABLE_POST_PROCESS" = "1" ] && EVAL_EXTRA_ARGS+=(--disable_post_process)

# --- 推理 ---
source /data/banlujie/sglang/bin/activate

SGLANG_DISABLE_CUDNN_CHECK=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="$SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK" \
CUDA_VISIBLE_DEVICES="$SGLANG_CUDA_VISIBLE_DEVICES" \
python3 infer_sglang.py \
    --model "$SGLANG_MODEL" \
    --tp-size "$SGLANG_TP_SIZE" \
    --dp-size "$SGLANG_DP_SIZE" \
    --gpu_utilization "$SGLANG_GPU_UTILIZATION" \
    --dataset "$DATASET" \
    --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation.txt" \
    --output_name "$OUTPUT_PREFIX" \
    --strategy "voting" \
    --num_candidates "$NUM_CANDIDATES" \
    --temperature 0.8 \
    --max_tokens "$SGLANG_MAX_TOKENS" \
    "${SGLANG_EXTRA_ARGS[@]}"

# --- 评估 ---
python3 eval_bird.py \
    --pred    "$PRED_FILE" \
    --gold    "$GOLD_FILE" \
    --db_path "$DB_ROOT_PATH" \
    --mode    major_voting \
    --num_cpus "$EVAL_NUM_CPUS" \
    --timeout  "$EVAL_TIMEOUT" \
    "${EVAL_EXTRA_ARGS[@]}"

echo ""
echo "=== Spider 投票策略评估完成 ==="
echo "数据集: $DATASET"
echo "结果文件: $PRED_FILE"
