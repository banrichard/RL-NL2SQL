#!/bin/bash
# 闭源 API 模型评估脚本，支持 BIRD / Spider 的 greedy 和 voting 评估。
#
# 用法:
#   bash scripts/eval_api.sh [MODEL] [DATASET] [STRATEGY] [NUM_CANDIDATES]
#
# DATASET 可选:
#   spider_test (默认), spider_dev, bird_dev, bird_train
#
# STRATEGY 可选:
#   greedy (默认), voting
#
# 环境变量覆盖:
#   BASE_URL, API_KEY, MAX_TOKENS, TEMPERATURE, CONCURRENCY
#   EVAL_DISABLE_POST_PROCESS (默认 1), STRATEGY, NUM_CANDIDATES

set -euo pipefail

MODEL="${1:-claude-sonnet-4-20250514}"
DATASET="${2:-bird_dev}"
STRATEGY="${3:-${STRATEGY:-voting}}"
NUM_CANDIDATES="${4:-${NUM_CANDIDATES:-4}}"
STRATEGY="${STRATEGY,,}"
MODEL_LOWER="${MODEL,,}"

case "$MODEL_LOWER" in
    sonnet-4.6|claude-sonnet-4.6)
        MODEL="claude-sonnet-4-6"
        MODEL_LOWER="${MODEL,,}"
        ;;
esac

DEFAULT_BASE_URL="https://api.vectorengine.ai/v1"
DEFAULT_API_KEY="sk-I6ivRQqABWjWDHcER2YcSSluSZySGnlqX8ZzCxXkxs6zsbHA"
MODEL_DEFAULT_CONCURRENCY="16"

case "$MODEL_LOWER" in
    gpt-5.4*)
        MODEL_DEFAULT_BASE_URL="https://tiantianapi.com/v1"
        MODEL_DEFAULT_API_KEY="sk-9c173126c34ac73be2df0371a05a5dc3feb79ba74f6071f4445b5e0ca29cab2d"
        ;;
    sonnet*|claude-sonnet*)
        MODEL_DEFAULT_BASE_URL="https://tiantianapi.com/v1"
        MODEL_DEFAULT_API_KEY="sk-1f24bd6e32f7a28d60ad9df3aa156b5e15071464a0b086480e72a72e4e130435"
        MODEL_DEFAULT_CONCURRENCY="4"
        ;;
    *)
        MODEL_DEFAULT_BASE_URL="$DEFAULT_BASE_URL"
        MODEL_DEFAULT_API_KEY="$DEFAULT_API_KEY"
        ;;
esac

BASE_URL="${BASE_URL:-$MODEL_DEFAULT_BASE_URL}"
API_KEY="${API_KEY:-$MODEL_DEFAULT_API_KEY}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
CONCURRENCY="${CONCURRENCY:-$MODEL_DEFAULT_CONCURRENCY}"
EVAL_NUM_CPUS="${EVAL_NUM_CPUS:-20}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30}"
EVAL_DISABLE_POST_PROCESS="${EVAL_DISABLE_POST_PROCESS:-1}"

case "$STRATEGY" in
    greedy)
        STRATEGY_TEMPERATURE_DEFAULT="1.0"
        OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_api_${DATASET}"
        EVAL_MODE="greedy_search"
        ;;
    voting)
        STRATEGY_TEMPERATURE_DEFAULT="1.0"
        OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_api_voting_${DATASET}"
        EVAL_MODE="major_voting"
        ;;
    *)
        echo "错误: 未知 STRATEGY='$STRATEGY'，可选值: greedy, voting"
        exit 1
        ;;
esac

TEMPERATURE="${TEMPERATURE:-$STRATEGY_TEMPERATURE_DEFAULT}"

MODEL_SLUG="${MODEL//\//_}"
MODEL_SLUG="${MODEL_SLUG//:/_}"
PRED_FILE="${OUTPUT_PREFIX}_${MODEL_SLUG}.json"

case "$DATASET" in
    bird_dev)
        GOLD_FILE="/data/banlujie/data/bird_dev/dev.json"
        DB_PATH="/data/banlujie/data/bird_dev/dev_databases"
        ;;
    bird_train)
        GOLD_FILE="/data/banlujie/data/bird_train/train/train.json"
        DB_PATH="/data/banlujie/data/bird_train/train/train_databases"
        ;;
    spider_dev)
        GOLD_FILE="/data/banlujie/data/spider/dev.json"
        DB_PATH="/data/banlujie/data/spider/database"
        ;;
    spider_test)
        GOLD_FILE="/data/banlujie/data/spider/test.json"
        DB_PATH="/data/banlujie/data/spider/test_database"
        ;;
    *)
        echo "错误: 未知 DATASET='$DATASET'，可选值: spider_dev, spider_test, bird_dev, bird_train"
        exit 1
        ;;
esac

echo "=== API Eval ==="
echo "Model                : $MODEL"
echo "Dataset              : $DATASET"
echo "Strategy             : $STRATEGY"
if [ "$STRATEGY" = "voting" ]; then
    echo "Num candidates       : $NUM_CANDIDATES"
fi
echo "Base URL             : $BASE_URL"
echo "Concurrency          : $CONCURRENCY"
echo "Max tokens           : $MAX_TOKENS"
echo "Temperature          : $TEMPERATURE"
echo "关闭 eval 后处理      : $EVAL_DISABLE_POST_PROCESS"
echo "Output               : $PRED_FILE"
echo ""

# Step 1: 推理
python3 infer_api.py \
    --base_url    "$BASE_URL" \
    --api_key     "$API_KEY" \
    --model       "$MODEL" \
    --dataset     "$DATASET" \
    --output_name "$OUTPUT_PREFIX" \
    --strategy    "$STRATEGY" \
    --num_candidates "$NUM_CANDIDATES" \
    --max_tokens  "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --concurrency "$CONCURRENCY"

# Step 2: 执行准确率
EVAL_EXTRA_ARGS=()
[ "$EVAL_DISABLE_POST_PROCESS" = "1" ] && EVAL_EXTRA_ARGS+=(--disable_post_process)

python3 eval_bird.py \
    --pred    "$PRED_FILE" \
    --gold    "$GOLD_FILE" \
    --db_path "$DB_PATH" \
    --mode    "$EVAL_MODE" \
    --num_cpus "$EVAL_NUM_CPUS" \
    --timeout  "$EVAL_TIMEOUT" \
    "${EVAL_EXTRA_ARGS[@]}"

echo ""
echo "=== Done ==="
echo "Dataset : $DATASET"
echo "Model   : $MODEL"
echo "Mode    : $EVAL_MODE"
echo "Pred    : $PRED_FILE"
