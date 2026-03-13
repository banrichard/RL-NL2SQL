#!/bin/bash

 deepspeed --module --include localhost:3,4,5,6 openrlhf.cli.train_sft \
   --save_path ./ckpt/qwen-2.5-7b-coder-instruct \
   --load_checkpoint \
   --max_len 4096 \
   --dataset /home/banlujie/data/synsql_5k/train.json \
   --eval_dataset /home/banlujie/data/synsql_5k/test.json\
   --input_key prompt \
   --output_key cot \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --pretrain Qwen/Qwen2.5-Coder-7B-Instruct \
   --packing_samples \
   --save_steps 100 \
   --logging_steps 10 \
   --eval_steps 100 \
   --zero_stage 2 \
   --max_epochs 2 \
   --bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb True \
   --wandb_project entropy_rl \
