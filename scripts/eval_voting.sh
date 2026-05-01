#!/bin/bash

# 投票策略评估脚本
# seeklhy/OmniSQL-7B
echo "=== 开始评估模型（投票策略）==="
MODEL_PATH="${1:-/home/banlujie/RL-NL2SQL/ckpt/Qwen3.5-9B}"
NUM_CANDIDATES="${2:-8}"
DATASET="${3:-bird_dev}"
OUTPUT_PREFIX="/data/banlujie/RL-NL2SQL/eval_res/eval_results_voting"
MODEL_NAME="$(basename "$MODEL_PATH")"
PRED_FILE="${OUTPUT_PREFIX}_${MODEL_NAME}.json"
BACKEND="${BACKEND:-sglang}"
SGLANG_MODEL="${SGLANG_MODEL:-$MODEL_PATH}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30}"
EVAL_NUM_CPUS="${EVAL_NUM_CPUS:-20}"
EVAL_DISABLE_POST_PROCESS="${EVAL_DISABLE_POST_PROCESS:-0}"
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-6,7}"
SGLANG_GPU_UTILIZATION="${SGLANG_GPU_UTILIZATION:-0.6}"
SGLANG_MAX_TOKENS="${SGLANG_MAX_TOKENS:-4096}"
SGLANG_BATCH_SIZE="${SGLANG_BATCH_SIZE:-}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-}"
SGLANG_QUANTIZATION_PARAM_PATH="${SGLANG_QUANTIZATION_PARAM_PATH:-}"
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-1}"
PARALLEL_MODE="${PARALLEL_MODE:-tp}"   # tp | dp | dp+tp

# --- GPU 数量 ---
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-6,7}"
IFS=',' read -r -a _SGLANG_GPU_ARRAY <<< "$SGLANG_CUDA_VISIBLE_DEVICES"
_GPU_COUNT="${#_SGLANG_GPU_ARRAY[@]}"
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
USE_ACCELERATE_FSDP="${USE_ACCELERATE_FSDP:-0}"
ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-2}"

if [ "$DATASET" = "bird_train" ]; then
    DEV_PATH="/data/banlujie/data/bird_train/train/train.json"
    TABLES_PATH="/data/banlujie/data/bird_train/train/train_tables.json"
    DB_ROOT_PATH="/data/banlujie/data/bird_train/train/train_databases"
    GOLD_FILE="$DEV_PATH"
    DB_EVAL_PATH="$DB_ROOT_PATH"
else
    DEV_PATH="/data/banlujie/data/bird_dev/dev.json"
    TABLES_PATH="/data/banlujie/data/bird_dev/dev_tables.json"
    DB_ROOT_PATH="/data/banlujie/data/bird_dev/dev_databases"
    GOLD_FILE="$DEV_PATH"
    DB_EVAL_PATH="$DB_ROOT_PATH"
fi

echo "使用模型: $MODEL_PATH"
echo "SGLang 实际推理模型: $SGLANG_MODEL"
if [ "$SGLANG_MODEL" != "$MODEL_PATH" ]; then
    echo "警告: SGLang 推理模型与评测模型路径不一致，结果可能与预期不符"
fi
echo "候选数量: $NUM_CANDIDATES"
echo "数据集: $DATASET"
echo "推理后端: $BACKEND"
echo "评估超时(秒): $EVAL_TIMEOUT"
echo "关闭 eval 后处理: $EVAL_DISABLE_POST_PROCESS"
echo "SGLang 可见GPU: $SGLANG_CUDA_VISIBLE_DEVICES"
echo "并行模式: $PARALLEL_MODE  (TP=${SGLANG_TP_SIZE}, DP=${SGLANG_DP_SIZE})"
echo "SGLang GPU 利用率: $SGLANG_GPU_UTILIZATION"
echo "SGLang Max Tokens: $SGLANG_MAX_TOKENS"
echo "SGLang Batch Size: ${SGLANG_BATCH_SIZE:-default}"
echo "SGLang KV Cache DType: ${SGLANG_KV_CACHE_DTYPE:-auto(default)}"
echo "SGLang Quant Param Path: ${SGLANG_QUANTIZATION_PARAM_PATH:-none}"
echo "SGLang TP 显存均衡检查: $SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"
echo "推理输出文件: $PRED_FILE"

EVAL_EXTRA_ARGS=()
if [ "$EVAL_DISABLE_POST_PROCESS" = "1" ]; then
    EVAL_EXTRA_ARGS+=(--disable_post_process)
fi

SGLANG_EXTRA_ARGS=()
if [ -n "$SGLANG_KV_CACHE_DTYPE" ]; then
    SGLANG_EXTRA_ARGS+=(--kv-cache-dtype "$SGLANG_KV_CACHE_DTYPE")
fi
if [ -n "$SGLANG_QUANTIZATION_PARAM_PATH" ]; then
    SGLANG_EXTRA_ARGS+=(--quantization-param-path "$SGLANG_QUANTIZATION_PARAM_PATH")
fi
if [ -n "$SGLANG_BATCH_SIZE" ]; then
    SGLANG_EXTRA_ARGS+=(--batch_size "$SGLANG_BATCH_SIZE")
fi

if [ "$BACKEND" = "sglang" ]; then
    source /data/banlujie/sglang/bin/activate
    SGLANG_DISABLE_CUDNN_CHECK=1 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="$SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK" CUDA_VISIBLE_DEVICES="$SGLANG_CUDA_VISIBLE_DEVICES" python3 infer_sglang.py \
        --model "$SGLANG_MODEL" \
        --tp-size "$SGLANG_TP_SIZE" \
        --dp-size "$SGLANG_DP_SIZE" \
        --gpu_utilization "$SGLANG_GPU_UTILIZATION" \
        --dataset "$DATASET" \
        --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
        --output_name "$OUTPUT_PREFIX" \
        --strategy "voting" \
        --num_candidates "$NUM_CANDIDATES" \
        --temperature 0.8 \
        --max_tokens "$SGLANG_MAX_TOKENS" \
        "${SGLANG_EXTRA_ARGS[@]}"
        # --system_prompt_path "/data/banlujie/RL-NL2SQL/templates/system_prompt_sql_expert.txt" \

elif [ "$BACKEND" = "transformers" ]; then
    if [ "$USE_ACCELERATE_FSDP" = "1" ]; then
        CUDA_VISIBLE_DEVICES=5,7 accelerate launch --num_processes "$ACCELERATE_NUM_PROCESSES" infer_transformers.py \
            --model_path "$MODEL_PATH" \
            --dev_path "$DEV_PATH" \
            --tables_path "$TABLES_PATH" \
            --db_root_path "$DB_ROOT_PATH" \
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
            --dev_path "$DEV_PATH" \
            --tables_path "$TABLES_PATH" \
            --db_root_path "$DB_ROOT_PATH" \
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
        --dev_path "$DEV_PATH" \
        --tables_path "$TABLES_PATH" \
        --db_root_path "$DB_ROOT_PATH" \
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
    --gold "$GOLD_FILE" \
    --db_path "$DB_EVAL_PATH" \
    --mode "major_voting" \
    --num_cpus "$EVAL_NUM_CPUS" \
    --timeout "$EVAL_TIMEOUT" \
    "${EVAL_EXTRA_ARGS[@]}"


echo "=== 投票策略评估完成 ==="
echo "结果来源文件: $PRED_FILE"
