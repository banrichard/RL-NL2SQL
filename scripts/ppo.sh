export RAY_local_fs_capacity_threshold=0.99
export REWARD_MODEL_PATH=/home/banlujie/RL-NL2SQL/ckpt/qwen3-0.6b-rm           
export REWARD_FUNC_AGGREGATION=mean     
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --vllm_gpu_memory_utilization 0.6 \
   --gamma 1.0 \
   --pretrain /home/banlujie/RL-NL2SQL/ckpt/Qwen3.5-9B \
   --remote_rm_url /data/banlujie/RL-NL2SQL/algorithm/reward_func.py \
   --save_path ./ckpt/qwen3.5-9b-ppo \
   --ckpt_path ./ckpt/ckpt/qwen3.5-9b-ppo-ckpt \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 16 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 8 \
   --n_samples_per_prompt 2 \
   --num_episodes 3 \
   --max_epochs 4 \
   --prompt_max_len 4096 \
   --max_samples 400000 \
   --generate_max_len 4096 \
   --zero_stage 2 \
   --param_dtype bf16 \
   --actor_learning_rate 2e-6 \
   --critic_learning_rate 5e-6 \
   --lr_warmup_ratio 0.1 \
   --init_kl_coef 0.001 \
   --prompt_data /data/banlujie/data/bird_dev/bird_dev_ppo.jsonl \
   --input_key prompt \
   --label_key sql \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --vllm_enable_sleep \
   --use_dynamic_batch \
   --train_max_tokens_per_gpu 8192 \
   --enable_vllm_is_correction \
   --overlap_comm \
   --enforce_eager \
   --use_wandb True \
   --adam_offload
   # --colocate_all_models \



   # --scm_num_nodes 1\
   # --scm_num_gpus_per_node 2 \
   # --scm_pretrain ./scm_decoder_trainer/final_checkpoint \
   # --enable_entropy_filtering \
   # --colocate_reward_critic_scm \
# SCM 相关参数（暂时禁用，先跑 pure PPO）
 #  --colocate_critic_reward \

