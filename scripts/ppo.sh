CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 train_ppo_ray.py \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --scm_num_nodes 1 \
   --scm_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.4 \
   --gamma 1.0 \
   --pretrain seeklhy/OmniSQL-7B \
   --reward_pretrain /home/banlujie/RL-NL2SQL/ckpt/omnisql-7b-rm \
   --save_path ./ckpt/omnisql-7b-ppo \
   --ckpt_path ./ckpt/ckpt/omnisql-7b-ppo \
   --save_hf_ckpt \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --num_episodes 3 \
   --max_epochs 4 \
   --prompt_max_len 4096 \
   --max_samples 400000 \
   --generate_max_len 8192 \
   --zero_stage 3 \
   --param_dtype bf16 \
   --actor_learning_rate 2e-6 \
   --critic_learning_rate 5e-6 \
   --lr_warmup_ratio 0.1 \
   --init_kl_coef 0.001 \
   --prompt_data /data/banlujie/data/synsql_5k/train.jsonl\
   --input_key prompt \
   --label_key sql \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --use_dynamic_batch \
   --scm_pretrain ./scm_decoder_trainer/final_checkpoint\
   --enable_entropy_filtering \
   --train_max_tokens_per_gpu 8192\
   --enable_vllm_is_correction \
   --colocate_reward_critic_scm \
   --colocate_actor_ref \
   --adam_offload \
   --use_wandb True\
      # 
   # --enable_entropy_filtering \
      # --colocate_actor_ref \
   #       --use_kl_loss \
   # --kl_estimator k3 \
      # --advantage_estimator reinforce \