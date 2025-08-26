#!/bin/bash

 deepspeed --module --include localhost:3,4,5,6 openrlhf.cli.train_sft \
   --save_path ./ckpt/phi4_mini_reasoning \
   --max_len 4096 \
   --dataset /home/banlujie/data/bird_train/train/train_sft.json \
   --eval_dataset /home/banlujie/data/bird_train/train/eval_sft.json\
   --input_key prompt \
   --output_key SQL \
   --apply_chat_template \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --max_samples 500000 \
   --pretrain microsoft/Phi-4-mini-reasoning \
   --packing_samples \
   --save_steps 40 \
   --logging_steps 10 \
   --eval_steps 40 \
   --zero_stage 2 \
   --max_epochs 20 \
   --bf16 \
   --flash_attn \
   --learning_rate 2.5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb True \
   --wandb_project entropy_rl \
