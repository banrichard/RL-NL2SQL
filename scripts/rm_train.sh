export HTTPS_PROXY="http://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"

deepspeed --module --include localhost:4,5,6,7 --master_port 29911 openrlhf.cli.train_rm \
   --save_path /data/banlujie/RL-NL2SQL/ckpt/qwen3-0.6b-rm \
   --save_steps 25 \
   --logging_steps 5 \
   --eval_steps 25 \
   --train_batch_size 256 \
   --micro_train_batch_size 8 \
   --pretrain Qwen/Qwen3-0.6B \
   --param_dtype bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --zero_stage 3 \
   --learning_rate 3e-6 \
   --lr_warmup_ratio 0.05 \
   --l2 0.01 \
   --loss sigmoid \
   --compute_fp32_loss \
   --dataset /data/banlujie/data/bird_dev/reward/v1_prm_train.jsonl \
   --eval_dataset /data/banlujie/data/bird_dev/reward/v1_prm_eval.jsonl \
   --chosen_key chosen \
   --rejected_key rejected \
   --attn_implementation flash_attention_2 \
   --packing_samples \
   --gradient_checkpointing \
   --use_wandb True \
   --wandb_project entropy_rl
   # --load_checkpoint \
