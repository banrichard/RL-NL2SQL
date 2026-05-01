#!/bin/bash
# Spider2-Lite API 评测脚本
#
# 用法:
#   bash scripts/eval_spider2_api.sh [MODEL] [EVAL_MODE]
#
#   MODEL:      API 模型名，默认 gemini-2.5-pro
#   EVAL_MODE:  local_only（默认）| bq | official
#               local_only  只评测 135 条 SQLite，无需 BigQuery 凭证
#               bq          SQLite + BigQuery（bq/ga 共 340 条），需凭证
#               official    调用官方 evaluate.py（等价于 bq）
#
# 环境变量:
#   BASE_URL      API base URL，默认 https://api.vectorengine.ai/v1
#   API_KEY       API 密钥（必填，或设 OPENAI_API_KEY）
#   MAX_TOKENS    默认 8192
#   TEMPERATURE   默认 0.0
#   CONCURRENCY   并发请求数，默认 16
#   FILTER_PREFIX 推理前缀过滤（默认与 EVAL_MODE 联动；all 则全量推理）
#   EVAL_MAX_WORKERS  评测并发线程数，默认 16
#   EVAL_TIMEOUT      SQLite 执行超时（秒），默认 60

set -euo pipefail

MODEL="${1:-gemini-2.5-pro}"
EVAL_MODE="${2:-local_only}"   # local_only | bq | official

BASE_URL="${BASE_URL:-https://api.vectorengine.ai/v1}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CONCURRENCY="${CONCURRENCY:-16}"

EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-16}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"

SPIDER2_DIR="/data/banlujie/data/spider2/spider2-lite"
EVAL_RES_DIR="/data/banlujie/RL-NL2SQL/eval_res/spider2"

if [ -z "$API_KEY" ]; then
    echo "错误: 请设置 API_KEY 环境变量或 OPENAI_API_KEY"
    exit 1
fi

# 推理前缀：local_only 只推 local，bq/official 推全量
if [ -n "${FILTER_PREFIX:-}" ]; then
    INFER_FILTER="$FILTER_PREFIX"
elif [ "$EVAL_MODE" = "local_only" ]; then
    INFER_FILTER="local"
else
    INFER_FILTER="all"
fi

MODEL_SLUG="${MODEL//\//_}"
MODEL_SLUG="${MODEL_SLUG//:/_}"
OUTPUT_PREFIX="${EVAL_RES_DIR}/spider2_api"
PRED_FILE="${OUTPUT_PREFIX}_${MODEL_SLUG}.json"

mkdir -p "$EVAL_RES_DIR"

echo "=== Spider2-Lite API 评测 ==="
echo "模型:         $MODEL"
echo "评测模式:     $EVAL_MODE"
echo "Base URL:     $BASE_URL"
echo "并发数:       $CONCURRENCY"
echo "最大 tokens:  $MAX_TOKENS"
echo "推理过滤:     $INFER_FILTER"
echo "输出 JSON:    $PRED_FILE"
echo ""

# ── Step 1: API 推理 ──────────────────────────────────────────────────────────
echo "--- Step 1: API 推理 ---"

FILTER_ARG=()
if [ "$INFER_FILTER" != "all" ]; then
    FILTER_ARG=(--filter_prefix "$INFER_FILTER")
fi

python3 infer_spider2_api.py \
    --base_url    "$BASE_URL" \
    --api_key     "$API_KEY" \
    --model       "$MODEL" \
    --spider2_dir "$SPIDER2_DIR" \
    --output_name "$OUTPUT_PREFIX" \
    --max_tokens  "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --concurrency "$CONCURRENCY" \
    "${FILTER_ARG[@]}"

echo ""
echo "推理完成: $PRED_FILE"

# ── Step 2: 评测 ─────────────────────────────────────────────────────────────
echo ""
echo "--- Step 2: 评测（模式: $EVAL_MODE）---"

python3 eval_spider2.py \
    --pred        "$PRED_FILE" \
    --mode        "$EVAL_MODE" \
    --spider2_dir "$SPIDER2_DIR" \
    --max_workers "$EVAL_MAX_WORKERS" \
    --timeout     "$EVAL_TIMEOUT"

echo ""
echo "=== 完成 ==="
echo "模型:       $MODEL"
echo "评测模式:   $EVAL_MODE"
echo "推理结果:   $PRED_FILE"
echo "评测结果:   ${PRED_FILE%.json}_eval.json"
echo "Summary:    eval_res/logs/eval_stats_spider2_spider2_api_${MODEL_SLUG}_*.log"
