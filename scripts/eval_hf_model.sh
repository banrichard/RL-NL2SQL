#!/bin/bash

# 基于 vLLM 的评估脚本（支持 Hugging Face 原始模型）
# 使用方法: ./eval_hf_model.sh [model_name]
# 例如: ./eval_hf_model.sh "Qwen/Qwen2.5-7B-Instruct"

# 默认模型（如果没有指定参数）
DEFAULT_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"

# 获取模型名称参数
if [ $# -eq 0 ]; then
    MODEL_PATH="$DEFAULT_MODEL"
    echo "未指定模型，使用默认模型: $MODEL_PATH"
else
    MODEL_PATH="$1"
    echo "使用指定模型: $MODEL_PATH"
fi

echo "=== 开始评估模型 ==="
echo "模型: $MODEL_PATH"
echo "注意：首次运行时会自动从 Hugging Face 下载模型，可能需要较长时间"
echo "确保网络连接正常且有足够的磁盘空间"

# 生成输出文件名（替换特殊字符）
OUTPUT_NAME=$(echo "$MODEL_PATH" | sed 's/\//_/g' | sed 's/-/_/g' | tr '[:upper:]' '[:lower:]')
OUTPUT_PATH="/home/banlujie/RL-NL2SQL/eval_results_${OUTPUT_NAME}.json"

python eval_vllm.py \
    --model_path "$MODEL_PATH" \
    --dev_path "/home/banlujie/data/bird_dev/dev.json" \
    --tables_path "/home/banlujie/data/bird_dev/dev_tables.json" \
    --db_root_path "/home/banlujie/data/bird_dev/dev_databases" \
    --template_path "/home/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
    --system_prompt_path "/home/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
    --output_path "$OUTPUT_PATH" \
    --temperature 0.1 \
    --max_tokens 16384 \
    --timeout 10

echo "=== 评估完成 ==="
echo "结果保存在: $OUTPUT_PATH"
