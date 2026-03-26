#!/bin/bash

# 对比评估脚本：同时运行贪心和投票策略

echo "=== 开始对比评估 ==="

MODEL_PATH="${1:-/data/banlujie/ckpt/Qwen3.5-9B}"
NUM_CANDIDATES="${2:-8}"
BACKEND="${BACKEND:-vllm}"
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:30000}"
SGLANG_API_KEY="${SGLANG_API_KEY:-EMPTY}"
SGLANG_MODEL="${SGLANG_MODEL:-$(basename "$MODEL_PATH")}" 
MODEL_NAME="$(basename "$MODEL_PATH")"
GREEDY_RESULT_PLAIN="/data/banlujie/RL-NL2SQL/eval_res/eval_results_greedy.json"
GREEDY_RESULT_WITH_MODEL="/data/banlujie/RL-NL2SQL/eval_res/eval_results_greedy_${MODEL_NAME}.json"
VOTING_RESULT_PLAIN="/data/banlujie/RL-NL2SQL/eval_res/eval_results_voting.json"
VOTING_RESULT_WITH_MODEL="/data/banlujie/RL-NL2SQL/eval_res/eval_results_voting_${MODEL_NAME}.json"

echo "模型: $MODEL_PATH"
echo "投票候选数: $NUM_CANDIDATES"
echo "推理后端: $BACKEND"
echo ""

# 运行贪心策略
echo "1. 运行贪心策略..."
BACKEND="$BACKEND" SGLANG_BASE_URL="$SGLANG_BASE_URL" SGLANG_API_KEY="$SGLANG_API_KEY" SGLANG_MODEL="$SGLANG_MODEL" \
./scripts/eval_greedy.sh "$MODEL_PATH"

echo ""
echo "等待5秒..."
sleep 5

# 运行投票策略
echo "2. 运行投票策略..."
BACKEND="$BACKEND" SGLANG_BASE_URL="$SGLANG_BASE_URL" SGLANG_API_KEY="$SGLANG_API_KEY" SGLANG_MODEL="$SGLANG_MODEL" \
./scripts/eval_voting.sh "$MODEL_PATH" "$NUM_CANDIDATES"

echo ""
echo "=== 对比评估完成 ==="
echo "贪心策略结果(固定文件): $GREEDY_RESULT_PLAIN"
echo "贪心策略结果(模型文件): $GREEDY_RESULT_WITH_MODEL"
echo "投票策略结果(固定文件): $VOTING_RESULT_PLAIN"
echo "投票策略结果(模型文件): $VOTING_RESULT_WITH_MODEL"

# 选择真实存在的结果文件路径（优先固定文件，其次模型文件）
if [ -f "$GREEDY_RESULT_PLAIN" ]; then
    GREEDY_RESULT_REAL="$GREEDY_RESULT_PLAIN"
elif [ -f "$GREEDY_RESULT_WITH_MODEL" ]; then
    GREEDY_RESULT_REAL="$GREEDY_RESULT_WITH_MODEL"
else
    GREEDY_RESULT_REAL=""
fi

if [ -f "$VOTING_RESULT_PLAIN" ]; then
    VOTING_RESULT_REAL="$VOTING_RESULT_PLAIN"
elif [ -f "$VOTING_RESULT_WITH_MODEL" ]; then
    VOTING_RESULT_REAL="$VOTING_RESULT_WITH_MODEL"
else
    VOTING_RESULT_REAL=""
fi

echo ""
echo "=== 结果对比 ==="
echo "贪心策略读取文件: ${GREEDY_RESULT_REAL:-未找到}"
echo "贪心策略准确率:"
python -c "
import json
with open('$GREEDY_RESULT_REAL', 'r') as f:
    data = json.load(f)
    print(f'  SQL提取率: {data[\"evaluation_stats\"][\"sql_extraction_rate\"]:.2%}')
    print(f'  执行准确率: {data[\"evaluation_stats\"][\"execution_accuracy\"]:.2%}')
" 2>/dev/null || echo "  贪心策略结果文件未找到"

echo "投票策略读取文件: ${VOTING_RESULT_REAL:-未找到}"
echo "投票策略准确率:"
python -c "
import json
with open('$VOTING_RESULT_REAL', 'r') as f:
    data = json.load(f)
    print(f'  SQL提取率: {data[\"evaluation_stats\"][\"sql_extraction_rate\"]:.2%}')
    print(f'  执行准确率: {data[\"evaluation_stats\"][\"execution_accuracy\"]:.2%}')
" 2>/dev/null || echo "  投票策略结果文件未找到"
