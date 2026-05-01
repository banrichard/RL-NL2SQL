#!/bin/bash

# Baseline 评测脚本：默认评测 SQL-R1-7B，并关闭 eval 阶段后处理
# 用法:
#   bash scripts/eval_baseline_sqlr1_7b.sh [DATASET]
#   DATASET 可选: bird_dev (默认), bird_train

set -euo pipefail

MODEL_PATH="/data/banlujie/models/SQL-R1-7B"
DATASET="${1:-bird_dev}"

echo "=== Baseline 评测：SQL-R1-7B（关闭 eval 后处理）==="
echo "模型: $MODEL_PATH"
echo "数据集: $DATASET"

EVAL_DISABLE_POST_PROCESS=1 bash scripts/eval_greedy.sh "$MODEL_PATH" "$DATASET"
