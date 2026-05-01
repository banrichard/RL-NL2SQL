#!/bin/bash

set -euo pipefail

echo "=== 开始评估模型（Voting，bird_train + synsql_5k_train）==="

MODEL_PATH="${1:-/home/banlujie/RL-NL2SQL/ckpt/Qwen3.5-9B}"
NUM_CANDIDATES="${2:-4}"
TARGET="${3:-synsql_5k_train}"   # both | bird_train | synsql_5k_train

BACKEND="${BACKEND:-sglang}"
SGLANG_MODEL="${SGLANG_MODEL:-$MODEL_PATH}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30}"
EVAL_NUM_CPUS="${EVAL_NUM_CPUS:-20}"
EVAL_DISABLE_POST_PROCESS="${EVAL_DISABLE_POST_PROCESS:-0}"

SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-2,4}"
SGLANG_GPU_UTILIZATION="${SGLANG_GPU_UTILIZATION:-0.6}"
SGLANG_MAX_TOKENS="${SGLANG_MAX_TOKENS:-4096}"
SGLANG_BATCH_SIZE="${SGLANG_BATCH_SIZE:-256}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-}"
SGLANG_QUANTIZATION_PARAM_PATH="${SGLANG_QUANTIZATION_PARAM_PATH:-}"
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-0}"
PARALLEL_MODE="${PARALLEL_MODE:-dp}"   # tp | dp | dp+tp

MODEL_NAME="$(basename "$MODEL_PATH")"

# --- GPU 数量 ---
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

SYN_SQL_TMP_GOLD=""
cleanup() {
    if [ -n "$SYN_SQL_TMP_GOLD" ] && [ -f "$SYN_SQL_TMP_GOLD" ]; then
        rm -f "$SYN_SQL_TMP_GOLD"
    fi
}
trap cleanup EXIT

run_one_dataset() {
    local DATASET_NAME="$1"
    local DEV_PATH="$2"
    local TABLES_PATH="$3"
    local DB_ROOT_PATH="$4"
    local OUTPUT_PREFIX="$5"
    local GOLD_FILE="$6"
    local DATASET_FOR_SGLANG="${7:-bird_train}"

    local PRED_FILE="${OUTPUT_PREFIX}_${MODEL_NAME}.json"

    echo "------------------------------------------------------------"
    echo "数据集: $DATASET_NAME"
    echo "模型: $MODEL_PATH"
    echo "SGLang 模型: $SGLANG_MODEL"
    echo "候选数: $NUM_CANDIDATES"
    echo "推理后端: $BACKEND"
    echo "DEV_PATH: $DEV_PATH"
    echo "TABLES_PATH: $TABLES_PATH"
    echo "DB_ROOT_PATH: $DB_ROOT_PATH"
    echo "GOLD_FILE: $GOLD_FILE"
    echo "输出: $PRED_FILE"
    echo "------------------------------------------------------------"

    if [ "$BACKEND" = "sglang" ]; then
        source /data/banlujie/sglang/bin/activate
        SGLANG_DISABLE_CUDNN_CHECK=1 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="$SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK" CUDA_VISIBLE_DEVICES="$SGLANG_CUDA_VISIBLE_DEVICES" python3 /data/banlujie/RL-NL2SQL/infer_sglang.py \
            --model "$SGLANG_MODEL" \
            --tp-size "$SGLANG_TP_SIZE" \
            --dp-size "$SGLANG_DP_SIZE" \
            --gpu_utilization "$SGLANG_GPU_UTILIZATION" \
            --dataset "$DATASET_FOR_SGLANG" \
            --dev_path "$DEV_PATH" \
            --tables_path "$TABLES_PATH" \
            --db_root_path "$DB_ROOT_PATH" \
            --template_path "/data/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt" \
            --output_name "$OUTPUT_PREFIX" \
            --strategy "voting" \
            --num_candidates "$NUM_CANDIDATES" \
            --temperature 0.8 \
            --max_tokens "$SGLANG_MAX_TOKENS" \
            "${SGLANG_EXTRA_ARGS[@]}"
    elif [ "$BACKEND" = "transformers" ]; then
        CUDA_VISIBLE_DEVICES="$SGLANG_CUDA_VISIBLE_DEVICES" python3 /data/banlujie/RL-NL2SQL/infer_transformers.py \
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
    else
        FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES="$SGLANG_CUDA_VISIBLE_DEVICES" python3 /data/banlujie/RL-NL2SQL/infer.py \
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

    python3 /data/banlujie/RL-NL2SQL/eval_bird.py \
        --pred "$PRED_FILE" \
        --gold "$GOLD_FILE" \
        --db_path "$DB_ROOT_PATH" \
        --mode "major_voting" \
        --num_cpus "$EVAL_NUM_CPUS" \
        --timeout "$EVAL_TIMEOUT" \
        "${EVAL_EXTRA_ARGS[@]}"
}

# 1) bird_train
if [ "$TARGET" = "both" ] || [ "$TARGET" = "bird_train" ]; then
    run_one_dataset \
        "bird_train" \
        "/data/banlujie/data/bird_train/train/train.json" \
        "/data/banlujie/data/bird_train/train/train_tables.json" \
        "/data/banlujie/data/bird_train/train/train_databases" \
        "/data/banlujie/RL-NL2SQL/eval_res/eval_results_voting_bird_train" \
        "/data/banlujie/data/bird_train/train/train.json" \
        "bird_train"
fi

# 2) synsql_5k_train
if [ "$TARGET" = "both" ] || [ "$TARGET" = "synsql_5k_train" ]; then
    SYN_SQL_TMP_GOLD="/tmp/synsql_5k_train_for_eval_${MODEL_NAME}_$$.json"
    python3 - <<'PY' > "$SYN_SQL_TMP_GOLD"
import json
in_path = "/data/banlujie/data/synsql_5k/train.jsonl"
rows = []
with open(in_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        sql = obj.get("SQL") or obj.get("sql") or obj.get("query") or ""
        rows.append({
            "question": obj.get("question", ""),
            "db_id": obj.get("db_id", ""),
            "query": sql,
            "SQL": sql
        })
print(json.dumps(rows, ensure_ascii=False))
PY

    run_one_dataset \
        "synsql_5k_train" \
        "/data/banlujie/data/synsql_5k/train.jsonl" \
        "/data/banlujie/data/synsql/tables.json" \
        "/data/banlujie/data/synsql/databases" \
        "/data/banlujie/RL-NL2SQL/eval_res/eval_results_voting_synsql_5k_train" \
        "$SYN_SQL_TMP_GOLD" \
        "bird_train"
fi

echo "=== Voting 双数据集评估完成 ==="
