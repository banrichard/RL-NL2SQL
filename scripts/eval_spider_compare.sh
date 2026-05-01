#!/bin/bash
# Spider 对比评估脚本：同时运行贪心和投票策略
# 完全对齐 eval_compare.sh 的结构与变量命名
#
# 用法:
#   bash scripts/eval_spider_compare.sh [MODEL_PATH] [NUM_CANDIDATES] [DATASET] [CUDA_DEVICES]
#
# 默认:
#   MODEL_PATH    = /data/banlujie/ckpt/Qwen3.5-9B
#   NUM_CANDIDATES= 8
#   DATASET       = spider_test
#   CUDA_DEVICES  = 6,7
#
# 并行模式（PARALLEL_MODE 环境变量）:
#   tp      (默认) 所有可见 GPU 做张量并行，每个请求跨全部 GPU
#           SGLANG_TP_SIZE 自动 = GPU 数量，SGLANG_DP_SIZE = 1
#
#   dp      所有可见 GPU 做数据并行，每个 GPU 独立运行一份模型
#           SGLANG_DP_SIZE 自动 = GPU 数量，SGLANG_TP_SIZE = 1
#           适合显存充足、想提升吞吐的场景
#
#   dp+tp   手动指定 SGLANG_DP_SIZE 和 SGLANG_TP_SIZE，
#           要求 DP_SIZE × TP_SIZE = 可见 GPU 数量
#           例如: 4 GPU, DP=2, TP=2
#
# 示例:
#   PARALLEL_MODE=dp bash scripts/eval_spider_compare.sh /path/to/model 8 spider_test 0,1,2,3
#   PARALLEL_MODE=dp+tp SGLANG_DP_SIZE=2 SGLANG_TP_SIZE=2 \
#       bash scripts/eval_spider_compare.sh /path/to/model 8 spider_test 0,1,2,3

set -euo pipefail

echo "=== 开始 Spider 对比评估 ==="

MODEL_PATH="${1:-/data/banlujie/ckpt/Qwen3.5-9B}"
NUM_CANDIDATES="${2:-8}"
DATASET="${3:-spider_test}"
CUDA_DEVICES="${4:-6,7}"

SGLANG_MODEL="${SGLANG_MODEL:-$MODEL_PATH}"
SGLANG_GPU_UTILIZATION="${SGLANG_GPU_UTILIZATION:-0.7}"
SGLANG_MAX_TOKENS="${SGLANG_MAX_TOKENS:-4096}"
SGLANG_BATCH_SIZE="${SGLANG_BATCH_SIZE:-512}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-}"
SGLANG_QUANTIZATION_PARAM_PATH="${SGLANG_QUANTIZATION_PARAM_PATH:-}"
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-0}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30}"
EVAL_NUM_CPUS="${EVAL_NUM_CPUS:-20}"
EVAL_DISABLE_POST_PROCESS="${EVAL_DISABLE_POST_PROCESS:-0}"
PARALLEL_MODE="${PARALLEL_MODE:-dp}"   # tp | dp | dp+tp

# --- 计算 GPU 数量 ---
IFS=',' read -r -a _GPU_ARRAY <<< "$CUDA_DEVICES"
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
        # 用户必须显式设置 SGLANG_DP_SIZE 和 SGLANG_TP_SIZE
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

MODEL_NAME="$(basename "$MODEL_PATH")"

if [ "$DATASET" = "spider_test" ]; then
    DB_ROOT_PATH="/data/banlujie/data/spider/test_database"
    GOLD_FILE="/data/banlujie/data/spider/test.json"
    GREEDY_OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_spider_greedy_test"
    VOTING_OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_spider_voting_test"
else
    DB_ROOT_PATH="/data/banlujie/data/spider/database"
    GOLD_FILE="/data/banlujie/data/spider/dev.json"
    GREEDY_OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_spider_greedy"
    VOTING_OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_spider_voting"
fi

GREEDY_RESULT_WITH_MODEL="${GREEDY_OUTPUT_PREFIX}_${MODEL_NAME}.json"
VOTING_RESULT_WITH_MODEL="${VOTING_OUTPUT_PREFIX}_${MODEL_NAME}.json"

echo "模型: $MODEL_PATH"
echo "SGLang 推理模型: $SGLANG_MODEL"
if [ "$SGLANG_MODEL" != "$MODEL_PATH" ]; then
    echo "警告: SGLang 推理模型与评测模型路径不一致，结果可能与预期不符"
fi
echo "投票候选数: $NUM_CANDIDATES"
echo "评估数据集: $DATASET"
echo "CUDA 设备: $CUDA_DEVICES (共 ${_GPU_COUNT} 张)"
echo "并行模式: $PARALLEL_MODE  (TP=${SGLANG_TP_SIZE}, DP=${SGLANG_DP_SIZE})"
echo "SGLang GPU 利用率: $SGLANG_GPU_UTILIZATION"
echo "SGLang Max Tokens: $SGLANG_MAX_TOKENS"
echo "SGLang Batch Size: $SGLANG_BATCH_SIZE"
echo "SGLang KV Cache DType: ${SGLANG_KV_CACHE_DTYPE:-auto(default)}"
echo "SGLang Quant Param Path: ${SGLANG_QUANTIZATION_PARAM_PATH:-none}"
echo "SGLang TP 显存均衡检查: $SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"
echo "评估超时: $EVAL_TIMEOUT"
echo "评估 CPU 数: $EVAL_NUM_CPUS"
echo "关闭 eval 后处理: $EVAL_DISABLE_POST_PROCESS"
echo ""

SGLANG_EXTRA_ARGS=()
[ -n "$SGLANG_KV_CACHE_DTYPE" ] && SGLANG_EXTRA_ARGS+=(--kv-cache-dtype "$SGLANG_KV_CACHE_DTYPE")
[ -n "$SGLANG_QUANTIZATION_PARAM_PATH" ] && SGLANG_EXTRA_ARGS+=(--quantization-param-path "$SGLANG_QUANTIZATION_PARAM_PATH")
[ -n "$SGLANG_BATCH_SIZE" ] && SGLANG_EXTRA_ARGS+=(--batch_size "$SGLANG_BATCH_SIZE")

EVAL_EXTRA_ARGS=()
[ "$EVAL_DISABLE_POST_PROCESS" = "1" ] && EVAL_EXTRA_ARGS+=(--disable_post_process)

source /data/banlujie/sglang/bin/activate

echo "1. 复用同一个 SGLang Engine 顺序运行 greedy 和 voting..."
SGLANG_DISABLE_CUDNN_CHECK=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="$SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK" \
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
python3 infer_sglang.py \
    --model "$SGLANG_MODEL" \
    --tp-size "$SGLANG_TP_SIZE" \
    --dp-size "$SGLANG_DP_SIZE" \
    --gpu_utilization "$SGLANG_GPU_UTILIZATION" \
    --dataset "$DATASET" \
    --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation.txt" \
    --strategy "compare" \
    --num_candidates "$NUM_CANDIDATES" \
    --max_tokens "$SGLANG_MAX_TOKENS" \
    --compare_greedy_output_name "$GREEDY_OUTPUT_PREFIX" \
    --compare_voting_output_name "$VOTING_OUTPUT_PREFIX" \
    "${SGLANG_EXTRA_ARGS[@]}"

echo ""
echo "2. 评估 greedy 结果..."
python3 eval_bird.py \
    --pred    "$GREEDY_RESULT_WITH_MODEL" \
    --gold    "$GOLD_FILE" \
    --db_path "$DB_ROOT_PATH" \
    --mode    greedy_search \
    --num_cpus "$EVAL_NUM_CPUS" \
    --timeout  "$EVAL_TIMEOUT" \
    "${EVAL_EXTRA_ARGS[@]}"

echo ""
echo "3. 评估 voting 结果..."
python3 eval_bird.py \
    --pred    "$VOTING_RESULT_WITH_MODEL" \
    --gold    "$GOLD_FILE" \
    --db_path "$DB_ROOT_PATH" \
    --mode    major_voting \
    --num_cpus "$EVAL_NUM_CPUS" \
    --timeout  "$EVAL_TIMEOUT" \
    "${EVAL_EXTRA_ARGS[@]}"

echo ""
echo "=== Spider 对比评估完成 ==="
echo "贪心策略结果: $GREEDY_RESULT_WITH_MODEL"
echo "投票策略结果: $VOTING_RESULT_WITH_MODEL"

echo ""
echo "=== 结果对比 ==="
for LABEL in "贪心策略" "投票策略"; do
    if [ "$LABEL" = "贪心策略" ]; then
        RESULT_FILE="$GREEDY_RESULT_WITH_MODEL"
    else
        RESULT_FILE="$VOTING_RESULT_WITH_MODEL"
    fi
    echo "${LABEL}读取文件: $RESULT_FILE"
    python3 -c "
import json, sys
path = '$RESULT_FILE'
try:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        total = len(data)
        extracted = sum(1 for x in data if any((s or '').strip() for s in x.get('pred_sqls', [])))
        print(f'  样本数: {total}')
        print(f'  SQL提取率(由pred_sqls统计): {extracted/total:.2%}')
        print('  执行准确率: 请以 eval_bird.py 输出为准')
    else:
        print('  无法识别结果文件格式')
except Exception as e:
    print(f'  结果文件不可读: {e}', file=sys.stderr)
" 2>/dev/null || echo "  结果文件不可读或不存在"
done
