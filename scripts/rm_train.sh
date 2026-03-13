export HTTPS_PROXY="http://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"

deepspeed --module --include localhost:0,2,3 openrlhf.cli.train_rm \
   --save_path /home/banlujie/RL-NL2SQL/ckpt/omnisql-7b-rm\
   --save_steps 500 \
   --logging_steps 50 \
   --eval_steps 500 \
   --train_batch_size 240 \
   --micro_train_batch_size 8 \
   --pretrain seeklhy/OmniSQL-7B \
   --bf16 \
   --max_epochs 3 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset /home/banlujie/data/synsql_5k/reward/reward_train_split.jsonl \
   --eval_dataset /home/banlujie/data/synsql_5k/reward/reward_eval_split.jsonl \
   --chosen_key chosen \
   --margin_loss\
   --rejected_key rejected \
   --attn_implementation flash_attention_2 \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing \
    --use_wandb True \
   --wandb_project entropy_rl