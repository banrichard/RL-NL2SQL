#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-wo_topology_pointwise}"
GPU="${2:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-8}"

case "$VARIANT" in
  wo_topology_pointwise)
    TRAIN_DATASET="/data/banlujie/data/bird_dev/reward_ablation/wo_topology_pointwise/train.jsonl"
    EVAL_DATASET="/data/banlujie/data/bird_dev/reward_ablation/wo_topology_pointwise/eval.jsonl"
    SAVE_PATH="/data/banlujie/RL-NL2SQL/ckpt/qwen3-0.6b-rm-wo-topology-pointwise"
    ;;
  exec_rm_pointwise)
    TRAIN_DATASET="/data/banlujie/data/bird_dev/reward_ablation/exec_rm_pointwise/train.jsonl"
    EVAL_DATASET="/data/banlujie/data/bird_dev/reward_ablation/exec_rm_pointwise/eval.jsonl"
    SAVE_PATH="/data/banlujie/RL-NL2SQL/ckpt/qwen3-0.6b-rm-exec-pointwise"
    ;;
  *)
    echo "Unsupported variant: $VARIANT" >&2
    echo "Use one of: wo_topology_pointwise, exec_rm_pointwise" >&2
    exit 1
    ;;
esac

cd /data/banlujie/RL-NL2SQL

RESUME_FROM_CHECKPOINT=""
LATEST_CKPT="$(find "$SAVE_PATH" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)"
if [[ -n "$LATEST_CKPT" ]]; then
  RESUME_FROM_CHECKPOINT="$LATEST_CKPT"
  echo "[resume] using checkpoint: $RESUME_FROM_CHECKPOINT"
fi

CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python scripts/train_pointwise_rm.py \
  --train-dataset "$TRAIN_DATASET" \
  --eval-dataset "$EVAL_DATASET" \
  --pretrain Qwen/Qwen3-0.6B \
  --save-path "$SAVE_PATH" \
  --max-len 4096 \
  --learning-rate 3e-6 \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM_STEPS" \
  --num-train-epochs 1 \
  --warmup-ratio 0.05 \
  --logging-steps "$LOGGING_STEPS" \
  --eval-steps "$EVAL_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --weight-decay 0.01 \
  --attn-implementation flash_attention_2 \
  --param-dtype bf16 \
  --gradient-checkpointing \
  --bf16 \
  --dataloader-num-workers "$NUM_WORKERS" \
  ${RESUME_FROM_CHECKPOINT:+--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT"} \
  --report-to none \
  --run-name "$VARIANT"
