#!/bin/bash

# 基于 vLLM 的评估脚本
# 使用默认参数评估 phi4_mini_reasoning 模型

echo "=== 开始评估 phi4_mini_reasoning 模型 ==="

python eval_vllm.py \
    --model_path "/home/banlujie/RL-NL2SQL/ckpt/phi4_mini_reasoning" \
    --dev_path "/home/banlujie/data/bird_dev/dev.json" \
    --tables_path "/home/banlujie/data/bird_dev/dev_tables.json" \
    --db_root_path "/home/banlujie/data/bird_dev/dev_databases" \
    --template_path "/home/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
    --system_prompt_path "/home/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
    --output_path "/home/banlujie/RL-NL2SQL/eval_results_phi4_mini.json" \
    --temperature 0.0 \
    --max_tokens 8192 \
    --timeout 10

echo "=== 评估完成 ==="
echo "结果保存在: /home/banlujie/RL-NL2SQL/eval_results_phi4_mini.json"
