#!/bin/bash
# 批量 Spider 评估脚本
# 对多个模型依次调用 eval_spider_compare.sh
#
# 用法:
#   bash scripts/eval_spider_batch.sh [DATASET] [CUDA_DEVICES] [NUM_CANDIDATES]
#
# 默认:
#   DATASET       = spider_dev
#   CUDA_DEVICES  = 6,7
#   NUM_CANDIDATES= 8

set -euo pipefail

DATASET="${1:-spider_dev}"
CUDA_DEVICES="${2:-6,7}"
NUM_CANDIDATES="${3:-8}"

# 后处理全部关闭，对齐 eval_compare.sh
EVAL_DISABLE_POST_PROCESS=1

LOG_DIR="/data/banlujie/RL-NL2SQL/eval_res/logs/spider_batch"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# DP 模式模型（7B/9B，显存充足，每张 GPU 独立跑一份）
DP_MODELS=(
    "/data/banlujie/models/Qwen/Qwen3.5-9B"
    "/data/banlujie/models/SQL-R1-7B"
    "/data/banlujie/models/XiyanSQL/XiyanSQL-7B"
    "/data/banlujie/models/OmniSQL/OmniSQL-7B"
)

# TP 模式模型（14B/15B，需要多卡合并显存）
TP_MODELS=(
    "/data/banlujie/models/OmniSQL/OmniSQL-14B"
    "/data/banlujie/models/CodeS-15B"
    "/data/banlujie/models/XiyanSQL/XiyanSQL-14B"
)

# ---- 统计 ----
TOTAL=$(( ${#DP_MODELS[@]} + ${#TP_MODELS[@]} ))
DONE=0
FAILED=()

run_model() {
    local model="$1"
    local mode="$2"
    local model_name
    model_name="$(basename "$model")"
    local log_file="${LOG_DIR}/${TIMESTAMP}_${model_name}.log"

    echo ""
    echo "########################################"
    echo "# [$((DONE+1))/$TOTAL] $model_name"
    echo "# 并行模式: $mode"
    echo "# 日志: $log_file"
    echo "########################################"

    if PARALLEL_MODE="$mode" \
       EVAL_DISABLE_POST_PROCESS="$EVAL_DISABLE_POST_PROCESS" \
       bash scripts/eval_spider_compare.sh \
           "$model" "$NUM_CANDIDATES" "$DATASET" "$CUDA_DEVICES" \
           2>&1 | tee "$log_file"; then
        echo "--- $model_name 完成 ---"
    else
        echo "--- $model_name 失败（exit $?），继续下一个 ---"
        FAILED+=("$model_name")
    fi

    DONE=$(( DONE + 1 ))
    echo ""
    echo "等待 10 秒，让 GPU 显存释放..."
    sleep 10
}

echo "========================================"
echo "=== Spider 批量评估 ==="
echo "数据集     : $DATASET"
echo "CUDA 设备  : $CUDA_DEVICES"
echo "投票候选数 : $NUM_CANDIDATES"
echo "关闭后处理 : $EVAL_DISABLE_POST_PROCESS"
echo "时间戳     : $TIMESTAMP"
echo "日志目录   : $LOG_DIR"
echo ""
echo "DP 模型 (${#DP_MODELS[@]} 个):"
for m in "${DP_MODELS[@]}"; do echo "  $m"; done
echo ""
echo "TP 模型 (${#TP_MODELS[@]} 个):"
for m in "${TP_MODELS[@]}"; do echo "  $m"; done
echo "========================================"

# ---- 依次评估 DP 模型 ----
for model in "${DP_MODELS[@]}"; do
    run_model "$model" "dp"
done

# ---- 依次评估 TP 模型 ----
for model in "${TP_MODELS[@]}"; do
    run_model "$model" "tp"
done

# ---- 汇总 ----
echo ""
echo "========================================"
echo "=== 批量评估完成 ==="
echo "共 $TOTAL 个模型，成功 $((TOTAL - ${#FAILED[@]})) 个"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "失败模型:"
    for m in "${FAILED[@]}"; do echo "  $m"; done
fi
echo ""
echo "=== 结果汇总 ==="
for model in "${DP_MODELS[@]}" "${TP_MODELS[@]}"; do
    model_name="$(basename "$model")"
    greedy_file="/data/banlujie/RL-NL2SQL/eval_res/eval_results_spider_greedy_${model_name}.json"
    voting_file="/data/banlujie/RL-NL2SQL/eval_res/eval_results_spider_voting_${model_name}.json"
    # 从日志中提取 EX Accuracy
    log_file="${LOG_DIR}/${TIMESTAMP}_${model_name}.log"
    echo "--- $model_name ---"
    if [ -f "$log_file" ]; then
        grep -E "EX Accuracy|ex_accuracy" "$log_file" | tail -4 || echo "  (日志中未找到准确率)"
    else
        echo "  (日志不存在)"
    fi
done
echo "========================================"
