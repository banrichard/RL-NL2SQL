#!/bin/bash

 deepspeed --module --include localhost:3,4,5,6 openrlhf.cli.train_sft \
   --save_path ../ckpt \
   --max_len 2048 \
   --dataset /home/banlujie/data/bird_train/train/train_clean.json \
   --input_key question \
   --output_key SQL \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain microsoft/Phi-4-mini-reasoning \
   --save_path ./checkpoint/phi4_mini_reasoning \
   --save_steps 1000 \
   --logging_steps 10 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 10 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing \
   --use_wandb False \
   --wandb_project entropy_rl \
#     --wandb 5c00be123cfc09b2a58a5d1071ffbfba53cd88d3
    # --packing_samples

