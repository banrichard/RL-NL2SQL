#!/bin/bash
set -euo pipefail
export CUDA_LAUNCH_BLOCKING=1
export MASTER_PORT=${MASTER_PORT:-29601}
# Use physical GPU IDs here (for example: DS_GPUS=4,5 ./scripts/sft.sh).
deepspeed --module --master_port "$MASTER_PORT" --include "localhost:5,7" openrlhf.cli.train_sft \
   --save_path ./ckpt/Qwen3.5-9B \
   --ckpt_path ./ckpt/checkpoints_sft_qwen3p5_9b \
   --max_len 8192 \
   --dataset /data/banlujie/data/synsql_5k/train.json \
   --eval_dataset /data/banlujie/data/synsql_5k/test.json \
   --input_key prompt \
   --output_key sql \
   --train_batch_size 64 \
   --micro_train_batch_size 2 \
   --pretrain /data/banlujie/models/Qwen/Qwen3.5-9B \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 2 \
   --max_epochs 3 \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 9e-7 \
   --gradient_checkpointing \
   --use_wandb True \
   --wandb_project entropy_rl \
   --adam_offload \
   --load_checkpoint \
   # --packing_samples \

