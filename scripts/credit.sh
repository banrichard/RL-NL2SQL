#!/bin/bash

deepspeed --include localhost:3,4,5,6 /home/banlujie/RL-NL2SQL/train_cm.py \
    --save_path "/home/banlujie/RL-NL2SQL/ckpt/phi4-7b-cm" \
    --pretrain "/home/banlujie/RL-NL2SQL/ckpt/phi4_mini_reasoning" \
    --dataset "/home/banlujie/data/bird_train/train/credit/credit_train_split.jsonl" \
    --eval_dataset "/home/banlujie/data/bird_train/train/credit/credit_eval_split.jsonl"\
    --save_steps 100 \
    --logging_steps 10 \
    --eval_steps 100 \
    --input_key prompt \
    --output_key target_credits \
    --train_batch_size 256 \
    --micro_train_batch_size 8 \
    --num_outputs 8 \
    --max_len 4096 \
    --max_epochs 15 \
    --learning_rate 1e-5 \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --load_checkpoint \
    --use_wandb True \
    --wandb_project entropy_rl\
    --gradient_checkpointing \
    --packing_samples
