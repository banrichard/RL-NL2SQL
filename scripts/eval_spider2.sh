#!/bin/bash
# Spider2-Lite 评测脚本（基于 SGLang 推理引擎）
#
# 用法:
#   bash scripts/eval_spider2.sh [MODEL_PATH] [EVAL_MODE]
#
#   EVAL_MODE（评测模式，默认 local_only）:
#     local_only  只评测 135 条 SQLite 样本，无需任何凭证
#     bq          SQLite + BigQuery（bq/ga 共 340 条），需 bigquery_credential.json
#     official    调用官方 evaluate.py（等价于 bq，sf_bq/* 官方也跳过）
#
# 数据库类型说明：
#   bq/*   180 条  BigQuery
#   ga/*    25 条  BigQuery
#   local/  135 条  SQLite（本地可执行）
#   sf_bq/  189 条  Snowflake 镜像 BigQuery 数据（官方 evaluate.py 也跳过）
#   sf/      18 条  同上
#
# 环境变量:
#   SGLANG_CUDA_VISIBLE_DEVICES  GPU 列表，默认 6,7
#   PARALLEL_MODE                tp | dp | dp+tp，默认 tp
#   SGLANG_TP_SIZE / SGLANG_DP_SIZE
#   SGLANG_GPU_UTILIZATION       默认 0.6
#   SGLANG_MAX_TOKENS            默认 8192
#   SGLANG_BATCH_SIZE            默认 100
#   EVAL_MAX_WORKERS             评测并发线程数，默认 16
#   EVAL_TIMEOUT                 SQL 执行超时（秒），默认 60

set -euo pipefail

echo "=== Spider2-Lite 评测（SGLang）==="

MODEL_PATH="${1:-/data/banlujie/ckpt/Qwen3.5-9B}"
EVAL_MODE="${2:-bq}"   # local_only | bq | official

SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-0,4}"
SGLANG_GPU_UTILIZATION="${SGLANG_GPU_UTILIZATION:-0.6}"
SGLANG_MAX_TOKENS="${SGLANG_MAX_TOKENS:-8192}"
SGLANG_BATCH_SIZE="${SGLANG_BATCH_SIZE:-100}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-}"
SGLANG_QUANTIZATION_PARAM_PATH="${SGLANG_QUANTIZATION_PARAM_PATH:-}"
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-0}"
PARALLEL_MODE="${PARALLEL_MODE:-tp}"

EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-16}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"

SPIDER2_DIR="/data/banlujie/data/spider2/spider2-lite"
EVAL_RES_DIR="/data/banlujie/RL-NL2SQL/eval_res/spider2"
TEMPLATE_PATH="/data/banlujie/RL-NL2SQL/templates/sql_generation_spider2.txt"

# 推理时的前缀过滤：bq 模式需要推理 bq/ga/local，local_only 只推理 local
if [ "$EVAL_MODE" = "local_only" ]; then
    INFER_FILTER="local"
else
    INFER_FILTER="all"   # bq/official 模式需要全量推理
fi

# --- GPU 数量与并行模式 ---
IFS=',' read -r -a _GPU_ARRAY <<< "$SGLANG_CUDA_VISIBLE_DEVICES"
_GPU_COUNT="${#_GPU_ARRAY[@]}"
[ "$_GPU_COUNT" -lt 1 ] && _GPU_COUNT=1

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
            echo "错误: DP(${SGLANG_DP_SIZE}) × TP(${SGLANG_TP_SIZE}) = ${_REQUIRED}，但可见 GPU = ${_GPU_COUNT}"
            exit 1
        fi
        ;;
    *)
        echo "错误: 未知 PARALLEL_MODE='$PARALLEL_MODE'（可选: tp | dp | dp+tp）"
        exit 1
        ;;
esac

MODEL_NAME="$(basename "$MODEL_PATH")"
OUTPUT_JSON="${EVAL_RES_DIR}/spider2_infer_${MODEL_NAME}.json"

echo "模型:          $MODEL_PATH"
echo "评测模式:      $EVAL_MODE"
echo "GPU:           $SGLANG_CUDA_VISIBLE_DEVICES  (${PARALLEL_MODE}: TP=${SGLANG_TP_SIZE}, DP=${SGLANG_DP_SIZE})"
echo "GPU 利用率:    $SGLANG_GPU_UTILIZATION"
echo "最大 tokens:   $SGLANG_MAX_TOKENS"
echo "批次大小:      $SGLANG_BATCH_SIZE"
echo "推理前缀过滤:  $INFER_FILTER"
echo "输出目录:      $EVAL_RES_DIR"
echo "推理 JSON:     $OUTPUT_JSON"
echo "TP 显存均衡检查: $SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"
echo ""

mkdir -p "$EVAL_RES_DIR"

# --- 额外 SGLang 参数 ---
SGLANG_EXTRA_ARGS=()
if [ -n "$SGLANG_KV_CACHE_DTYPE" ]; then
    SGLANG_EXTRA_ARGS+=(--kv-cache-dtype "$SGLANG_KV_CACHE_DTYPE")
fi
if [ -n "$SGLANG_QUANTIZATION_PARAM_PATH" ]; then
    SGLANG_EXTRA_ARGS+=(--quantization-param-path "$SGLANG_QUANTIZATION_PARAM_PATH")
fi

# --- 推理前缀参数 ---
FILTER_ARG=()
if [ "$INFER_FILTER" != "all" ]; then
    FILTER_ARG=(--filter_prefix "$INFER_FILTER")
fi

# ── Step 1: 推理 ─────────────────────────────────────────────────────────────
echo "--- Step 1: SGLang 推理 ---"
source /data/banlujie/sglang/bin/activate

SGLANG_DISABLE_CUDNN_CHECK=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="$SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK" \
CUDA_VISIBLE_DEVICES="$SGLANG_CUDA_VISIBLE_DEVICES" \
python3 infer_spider2.py \
    --model "$MODEL_PATH" \
    --tp-size "$SGLANG_TP_SIZE" \
    --dp-size "$SGLANG_DP_SIZE" \
    --gpu_utilization "$SGLANG_GPU_UTILIZATION" \
    --spider2_dir "$SPIDER2_DIR" \
    --template_path "$TEMPLATE_PATH" \
    --output_dir "$EVAL_RES_DIR" \
    --output_name "$OUTPUT_JSON" \
    --strategy greedy \
    --temperature 0.0 \
    --max_tokens "$SGLANG_MAX_TOKENS" \
    --batch_size "$SGLANG_BATCH_SIZE" \
    "${FILTER_ARG[@]}" \
    "${SGLANG_EXTRA_ARGS[@]}"

echo ""
echo "推理完成，结果: $OUTPUT_JSON"

# ── Step 2: 评测 ─────────────────────────────────────────────────────────────
echo ""
echo "--- Step 2: 执行评测（模式: $EVAL_MODE）---"

python3 eval_spider2.py \
    --pred "$OUTPUT_JSON" \
    --mode "$EVAL_MODE" \
    --spider2_dir "$SPIDER2_DIR" \
    --max_workers "$EVAL_MAX_WORKERS" \
    --timeout "$EVAL_TIMEOUT"

echo ""
echo "=== Spider2-Lite 评测完成 ==="
echo "模型:       $MODEL_PATH"
echo "评测模式:   $EVAL_MODE"
echo "推理结果:   $OUTPUT_JSON"
echo "评测结果:   ${OUTPUT_JSON%.json}_eval.json"
echo "Summary:    eval_res/logs/eval_stats_spider2_spider2_infer_${MODEL_NAME}_*.log"
