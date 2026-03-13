#!/bin/bash

# 对比评估脚本：同时运行贪心和投票策略

echo "=== 开始对比评估 ==="

MODEL_PATH="${1:-microsoft/Phi-4-mini-reasoning}"
NUM_CANDIDATES="${2:-8}"

echo "模型: $MODEL_PATH"
echo "投票候选数: $NUM_CANDIDATES"
echo ""

# 运行贪心策略
echo "1. 运行贪心策略..."
./scripts/eval_greedy.sh "$MODEL_PATH"

echo ""
echo "等待5秒..."
sleep 5

# 运行投票策略
echo "2. 运行投票策略..."
./scripts/eval_voting.sh "$MODEL_PATH" "$NUM_CANDIDATES"

echo ""
echo "=== 对比评估完成 ==="
echo "贪心策略结果: /home/banlujie/RL-NL2SQL/eval_results_greedy.json"
echo "投票策略结果: /home/banlujie/RL-NL2SQL/eval_results_voting.json"

# 简单对比输出
echo ""
echo "=== 结果对比 ==="
echo "贪心策略准确率:"
python -c "
import json
with open('/home/banlujie/RL-NL2SQL/eval_results_greedy.json', 'r') as f:
    data = json.load(f)
    print(f'  SQL提取率: {data[\"evaluation_stats\"][\"sql_extraction_rate\"]:.2%}')
    print(f'  执行准确率: {data[\"evaluation_stats\"][\"execution_accuracy\"]:.2%}')
" 2>/dev/null || echo "  贪心策略结果文件未找到"

echo "投票策略准确率:"
python -c "
import json
with open('/home/banlujie/RL-NL2SQL/eval_results_voting.json', 'r') as f:
    data = json.load(f)
    print(f'  SQL提取率: {data[\"evaluation_stats\"][\"sql_extraction_rate\"]:.2%}')
    print(f'  执行准确率: {data[\"evaluation_stats\"][\"execution_accuracy\"]:.2%}')
" 2>/dev/null || echo "  投票策略结果文件未找到"
