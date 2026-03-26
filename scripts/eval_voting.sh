#!/bin/bash

# 投票策略评估脚本
# seeklhy/OmniSQL-7B
echo "=== 开始评估模型（投票策略）==="
MODEL_PATH="${1:-/home/banlujie/RL-NL2SQL/ckpt/Qwen3.5-9B}"
NUM_CANDIDATES="${2:-8}"
OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_voting"
MODEL_NAME="$(basename "$MODEL_PATH")"
PRED_FILE="${OUTPUT_PREFIX}_${MODEL_NAME}.json"
BACKEND="${BACKEND:-sglang}"
SGLANG_MODEL="${SGLANG_MODEL:-$MODEL_PATH}"
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-1,3}"
if [ -z "${SGLANG_TP_SIZE:-}" ]; then
    _SGLANG_GPU_LIST="${SGLANG_CUDA_VISIBLE_DEVICES// /}"
    IFS=',' read -r -a _SGLANG_GPU_ARRAY <<< "$_SGLANG_GPU_LIST"
    SGLANG_TP_SIZE="${#_SGLANG_GPU_ARRAY[@]}"
    if [ -z "$SGLANG_TP_SIZE" ] || [ "$SGLANG_TP_SIZE" -lt 1 ]; then
        SGLANG_TP_SIZE=1
    fi
fi
_SGLANG_GPU_LIST="${SGLANG_CUDA_VISIBLE_DEVICES// /}"
IFS=',' read -r -a _SGLANG_GPU_ARRAY <<< "$_SGLANG_GPU_LIST"
SGLANG_VISIBLE_GPU_COUNT="${#_SGLANG_GPU_ARRAY[@]}"
if [ "$SGLANG_TP_SIZE" -ne "$SGLANG_VISIBLE_GPU_COUNT" ]; then
    echo "警告: SGLANG_TP_SIZE($SGLANG_TP_SIZE) 与可见GPU数量($SGLANG_VISIBLE_GPU_COUNT)不一致，已自动修正"
    SGLANG_TP_SIZE="$SGLANG_VISIBLE_GPU_COUNT"
fi
USE_ACCELERATE_FSDP="${USE_ACCELERATE_FSDP:-0}"
ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-2}"
echo "使用模型: $MODEL_PATH"
echo "候选数量: $NUM_CANDIDATES"
echo "推理后端: $BACKEND"
echo "SGLang 可见GPU: $SGLANG_CUDA_VISIBLE_DEVICES"
echo "SGLang Tensor Parallel: $SGLANG_TP_SIZE"
echo "推理输出文件: $PRED_FILE"

if [ "$BACKEND" = "sglang" ]; then
SGLANG_DISABLE_CUDNN_CHECK=1 CUDA_VISIBLE_DEVICES="$SGLANG_CUDA_VISIBLE_DEVICES" python3 infer_sglang.py \
        --model "$SGLANG_MODEL" \
    --dp-size "$SGLANG_TP_SIZE" \
        --dev_path "/data/banlujie/data/bird_dev/dev_omni.json" \
        --tables_path "/data/banlujie/data/bird_dev/dev_tables.json" \
        --db_root_path "/data/banlujie/data/bird_dev/dev_databases" \
        --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
        --output_name "$OUTPUT_PREFIX" \
        --strategy "voting" \
        --num_candidates "$NUM_CANDIDATES" \
        --temperature 0.8 \
        --max_tokens 4096
        # --system_prompt_path "/data/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \

elif [ "$BACKEND" = "transformers" ]; then
    if [ "$USE_ACCELERATE_FSDP" = "1" ]; then
        CUDA_VISIBLE_DEVICES=5,7 accelerate launch --num_processes "$ACCELERATE_NUM_PROCESSES" infer_transformers.py \
            --model_path "$MODEL_PATH" \
            --dev_path "/data/banlujie/data/bird_dev/dev_omni.json" \
            --tables_path "/data/banlujie/data/bird_dev/dev_tables.json" \
            --db_root_path "/data/banlujie/data/bird_dev/dev_databases" \
            --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
            --system_prompt_path "/data/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
            --output_name "$OUTPUT_PREFIX" \
            --strategy "voting" \
            --num_candidates "$NUM_CANDIDATES" \
            --temperature 0.8 \
            --max_tokens 2048 \
            --batch_size 8 \
            --torch_dtype bfloat16 \
            --use_accelerate_fsdp
    else
        CUDA_VISIBLE_DEVICES=5,7 python3 infer_transformers.py \
            --model_path "$MODEL_PATH" \
            --dev_path "/data/banlujie/data/bird_dev/dev_omni.json" \
            --tables_path "/data/banlujie/data/bird_dev/dev_tables.json" \
            --db_root_path "/data/banlujie/data/bird_dev/dev_databases" \
            --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
            --system_prompt_path "/data/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
            --output_name "$OUTPUT_PREFIX" \
            --strategy "voting" \
            --num_candidates "$NUM_CANDIDATES" \
            --temperature 0.8 \
            --max_tokens 2048 \
            --batch_size 32 \
            --torch_dtype bfloat16
    fi
else
   FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=1,3,5,7 python3 infer.py \
        --model_path "$MODEL_PATH" \
        --dev_path "/data/banlujie/data/bird_dev/dev_omni.json" \
        --tables_path "/data/banlujie/data/bird_dev/dev_tables.json" \
        --db_root_path "/data/banlujie/data/bird_dev/dev_databases" \
        --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
        --system_prompt_path "/data/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \
        --output_name "$OUTPUT_PREFIX" \
        --strategy "voting" \
        --num_candidates "$NUM_CANDIDATES" \
        --temperature 0.8 \
        --max_tokens 4096
fi
    
python3 eval_bird.py \
    --pred "$PRED_FILE" \
    --mode "major_voting"


echo "=== 投票策略评估完成 ==="
echo "结果来源文件: $PRED_FILE"
