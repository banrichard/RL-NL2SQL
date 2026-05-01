#!/bin/bash

# 对比评估脚本：同时运行贪心和投票策略

echo "=== 开始对比评估 ==="
source /data/banlujie/sglang/bin/activate
MODEL_PATH="${1:-/data/banlujie/models/XiyanSQL/XiyanSQL-7B}"
NUM_CANDIDATES="${2:-8}"
DATASET="${3:-bird_dev}"
CUDA_DEVICES="${4:-0,4}"
BACKEND="${BACKEND:-sglang}"
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:30000}"
SGLANG_API_KEY="${SGLANG_API_KEY:-EMPTY}"
SGLANG_MODEL="${SGLANG_MODEL:-$MODEL_PATH}" 
SGLANG_GPU_UTILIZATION="${SGLANG_GPU_UTILIZATION:-0.45}"
SGLANG_MAX_TOKENS="${SGLANG_MAX_TOKENS:-4096}"
SGLANG_BATCH_SIZE="${SGLANG_BATCH_SIZE:-512}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-}"
SGLANG_QUANTIZATION_PARAM_PATH="${SGLANG_QUANTIZATION_PARAM_PATH:-}"
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-0}"
EVAL_DISABLE_POST_PROCESS="${EVAL_DISABLE_POST_PROCESS:-1}"
PARALLEL_MODE="${PARALLEL_MODE:-tp}"   # tp | dp | dp+tp

# --- GPU 数量 ---
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
GREEDY_RESULT_PLAIN="/data/banlujie/RL-NL2SQL/eval_res/eval_results_greedy.json"
GREEDY_RESULT_WITH_MODEL="/data/banlujie/RL-NL2SQL/eval_res/eval_results_greedy_${MODEL_NAME}.json"
VOTING_RESULT_PLAIN="/data/banlujie/RL-NL2SQL/eval_res/eval_results_voting.json"
VOTING_RESULT_WITH_MODEL="/data/banlujie/RL-NL2SQL/eval_res/eval_results_voting_${MODEL_NAME}.json"

echo "模型: $MODEL_PATH"
echo "SGLang 实际推理模型: $SGLANG_MODEL"
if [ "$SGLANG_MODEL" != "$MODEL_PATH" ]; then
    echo "警告: SGLang 推理模型与评测模型路径不一致，结果可能与预期不符"
fi
echo "投票候选数: $NUM_CANDIDATES"
echo "评估数据集: $DATASET"
echo "推理后端: $BACKEND"
echo "CUDA 设备: $CUDA_DEVICES"
echo "并行模式: $PARALLEL_MODE  (TP=${SGLANG_TP_SIZE}, DP=${SGLANG_DP_SIZE})"
echo "SGLang GPU 利用率: $SGLANG_GPU_UTILIZATION"
echo "SGLang Max Tokens: $SGLANG_MAX_TOKENS"
echo "SGLang Batch Size: $SGLANG_BATCH_SIZE"
echo "SGLang KV Cache DType: ${SGLANG_KV_CACHE_DTYPE:-auto(default)}"
echo "SGLang Quant Param Path: ${SGLANG_QUANTIZATION_PARAM_PATH:-none}"
echo "SGLang TP 显存均衡检查: $SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"
echo "关闭 eval 后处理: $EVAL_DISABLE_POST_PROCESS"
echo ""

# 公共环境变量，透传给子脚本
_COMMON_ENV=(
    env
    BACKEND="$BACKEND"
    SGLANG_BASE_URL="$SGLANG_BASE_URL"
    SGLANG_API_KEY="$SGLANG_API_KEY"
    SGLANG_MODEL="$SGLANG_MODEL"
    SGLANG_CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    SGLANG_TP_SIZE="$SGLANG_TP_SIZE"
    SGLANG_DP_SIZE="$SGLANG_DP_SIZE"
    PARALLEL_MODE="$PARALLEL_MODE"
    SGLANG_GPU_UTILIZATION="$SGLANG_GPU_UTILIZATION"
    SGLANG_MAX_TOKENS="$SGLANG_MAX_TOKENS"
    SGLANG_BATCH_SIZE="$SGLANG_BATCH_SIZE"
    SGLANG_KV_CACHE_DTYPE="$SGLANG_KV_CACHE_DTYPE"
    SGLANG_QUANTIZATION_PARAM_PATH="$SGLANG_QUANTIZATION_PARAM_PATH"
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="$SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"
    EVAL_DISABLE_POST_PROCESS="$EVAL_DISABLE_POST_PROCESS"
)

# 运行贪心策略
echo "1. 运行贪心策略..."
"${_COMMON_ENV[@]}" ./scripts/eval_greedy.sh "$MODEL_PATH" "$DATASET"

echo ""
echo "等待5秒..."
sleep 5

# 运行投票策略
echo "2. 运行投票策略..."
"${_COMMON_ENV[@]}" ./scripts/eval_voting.sh "$MODEL_PATH" "$NUM_CANDIDATES" "$DATASET"

echo ""
echo "=== 对比评估完成 ==="
echo "贪心策略结果(固定文件): $GREEDY_RESULT_PLAIN"
echo "贪心策略结果(模型文件): $GREEDY_RESULT_WITH_MODEL"
echo "投票策略结果(固定文件): $VOTING_RESULT_PLAIN"
echo "投票策略结果(模型文件): $VOTING_RESULT_WITH_MODEL"

# 选择真实存在的结果文件路径（优先本次模型文件，其次固定文件）
if [ -f "$GREEDY_RESULT_WITH_MODEL" ]; then
    GREEDY_RESULT_REAL="$GREEDY_RESULT_WITH_MODEL"
elif [ -f "$GREEDY_RESULT_PLAIN" ]; then
    GREEDY_RESULT_REAL="$GREEDY_RESULT_PLAIN"
else
    GREEDY_RESULT_REAL=""
fi

if [ -f "$VOTING_RESULT_WITH_MODEL" ]; then
    VOTING_RESULT_REAL="$VOTING_RESULT_WITH_MODEL"
elif [ -f "$VOTING_RESULT_PLAIN" ]; then
    VOTING_RESULT_REAL="$VOTING_RESULT_PLAIN"
else
    VOTING_RESULT_REAL=""
fi

echo ""
echo "=== 结果对比 ==="
echo "贪心策略读取文件: ${GREEDY_RESULT_REAL:-未找到}"
echo "贪心策略结果摘要:"
python -c "
import json
path = '$GREEDY_RESULT_REAL'
if not path:
    raise FileNotFoundError('empty path')
with open(path, 'r') as f:
    data = json.load(f)
if isinstance(data, dict) and 'evaluation_stats' in data:
    print(f'  SQL提取率: {data[\"evaluation_stats\"].get(\"sql_extraction_rate\", 0):.2%}')
    print(f'  执行准确率: {data[\"evaluation_stats\"].get(\"execution_accuracy\", 0):.2%}')
elif isinstance(data, list):
    total = len(data)
    extracted = sum(1 for x in data if any((s or '').strip() for s in x.get('pred_sqls', [])))
    rate = extracted / total if total else 0
    print(f'  样本数: {total}')
    print(f'  SQL提取率(由pred_sqls统计): {rate:.2%}')
    print('  执行准确率: 该文件不包含 evaluation_stats（请以 eval_bird.py 输出为准）')
else:
    print('  无法识别结果文件格式')
" 2>/dev/null || echo "  贪心策略结果文件不可读或不存在"

echo "投票策略读取文件: ${VOTING_RESULT_REAL:-未找到}"
echo "投票策略结果摘要:"
python -c "
import json
path = '$VOTING_RESULT_REAL'
if not path:
    raise FileNotFoundError('empty path')
with open(path, 'r') as f:
    data = json.load(f)
if isinstance(data, dict) and 'evaluation_stats' in data:
    print(f'  SQL提取率: {data[\"evaluation_stats\"].get(\"sql_extraction_rate\", 0):.2%}')
    print(f'  执行准确率: {data[\"evaluation_stats\"].get(\"execution_accuracy\", 0):.2%}')
elif isinstance(data, list):
    total = len(data)
    extracted = sum(1 for x in data if any((s or '').strip() for s in x.get('pred_sqls', [])))
    rate = extracted / total if total else 0
    print(f'  样本数: {total}')
    print(f'  SQL提取率(由pred_sqls统计): {rate:.2%}')
    print('  执行准确率: 该文件不包含 evaluation_stats（请以 eval_bird.py 输出为准）')
else:
    print('  无法识别结果文件格式')
" 2>/dev/null || echo "  投票策略结果文件不可读或不存在"
