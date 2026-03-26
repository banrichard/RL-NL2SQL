#!/bin/bash

# 使用 LoRA adapter 和 vLLM 进行评估的示例脚本

# 基础模型和 LoRA adapter 路径
BASE_MODEL="/data/banlujie/models/Qwen/Qwen3.5-9B"
LORA_ADAPTER="/home/banlujie/RL-NL2SQL/ckpt/Qwen3.5-9B-SFT"

# 运行推理
CUDA_VISIBLE_DEVICES=4 python infer.py \
    --base_model_path "${BASE_MODEL}" \
    --lora_path "${LORA_ADAPTER}" \
    --dev_path "/data/banlujie/data/bird_dev/dev.json" \
    --tables_path "/data/banlujie/data/bird_dev/dev_tables.json" \
    --db_root_path "/data/banlujie/data/bird_dev/dev_databases" \
    --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation.txt" \
    --strategy greedy \
    --temperature 0.0 \
    --max_tokens 4096 \
    --gpu_utilization 0.8 \
    --output_name "/data/banlujie/RL-NL2SQL/eval_res/lora_inference_result"
