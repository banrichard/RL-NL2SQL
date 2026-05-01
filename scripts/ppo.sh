export RAY_local_fs_capacity_threshold=0.99
export RAY_memory_usage_threshold=0.99
export REWARD_MODEL_PATH=/home/banlujie/RL-NL2SQL/ckpt/qwen3-0.6b-rm
export REWARD_FUNC_AGGREGATION=mean
export REMOTE_RM_URL_FOR_AGENT=/data/banlujie/RL-NL2SQL/algorithm/reward_func.py
# Reward model device inside worker process. Keep cuda:0 so it maps to
# the GPU assigned to that Ray worker.
export REWARD_MODEL_DEVICE="${REWARD_MODEL_DEVICE:-cuda:0}"
# Reward model memory is controlled independently from --vllm_gpu_memory_utilization.
export REWARD_MODEL_MAX_MEMORY="${REWARD_MODEL_MAX_MEMORY:-}"
export REWARD_MODEL_GPU_MEMORY_UTILIZATION="${REWARD_MODEL_GPU_MEMORY_UTILIZATION:-0.12}"
export VLLM_THINKING_TOKEN_BUDGET="${VLLM_THINKING_TOKEN_BUDGET:-512}"
export VLLM_ANSWER_TOKEN_RESERVE="${VLLM_ANSWER_TOKEN_RESERVE:-512}"
export VLLM_STOP_SEQUENCES="${VLLM_STOP_SEQUENCES:-</answer>,<|im_end|>,</s>}"
export VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
export PPO_TEMPERATURE="${PPO_TEMPERATURE:-0.6}"
export PPO_MAX_EPOCHS="${PPO_MAX_EPOCHS:-1}"
export PPO_MAX_NEW_TOKENS="${PPO_MAX_NEW_TOKENS:-2048}"
export PPO_N_SAMPLES_PER_PROMPT="${PPO_N_SAMPLES_PER_PROMPT:-2}"
export PPO_ACTOR_LR="${PPO_ACTOR_LR:-1e-6}"
export PPO_INIT_KL_COEF="${PPO_INIT_KL_COEF:-0.01}"
export PPO_KL_TARGET="${PPO_KL_TARGET:-0.1}"
export PPO_ASYNC_TRAIN="${PPO_ASYNC_TRAIN:-1}"
export PPO_ASYNC_QUEUE_SIZE="${PPO_ASYNC_QUEUE_SIZE:-2}"
export PPO_PARTIAL_ROLLOUT="${PPO_PARTIAL_ROLLOUT:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,5,6,7}"

ASYNC_ARGS=()
if [ "$PPO_ASYNC_TRAIN" = "1" ]; then
  ASYNC_ARGS+=(--async_train --async_queue_size "$PPO_ASYNC_QUEUE_SIZE")
  if [ "$PPO_PARTIAL_ROLLOUT" = "1" ]; then
    ASYNC_ARGS+=(--partial_rollout)
  fi
fi

# PPO with critic on 4 GPUs:
#   - actor/ref/critic/vLLM each uses 1 GPU
#   - disable colocate_all_models to avoid stacking model memory on one card
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 2\
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.45\
   --vllm_reasoning_parser "$VLLM_REASONING_PARSER" \
   --colocate_actor_ref \
   --gamma 1.0 \
   --temperature "$PPO_TEMPERATURE" \
   --use_kl_loss \
   --kl_estimator k3 \
   --kl_target "$PPO_KL_TARGET" \
   --pretrain /home/banlujie/RL-NL2SQL/ckpt/Qwen3.5-9B \
   --remote_rm_url /data/banlujie/RL-NL2SQL/algorithm/reward_func.py \
   --agent_func_path /data/banlujie/RL-NL2SQL/algorithm/agent_executor_with_thinking_budget.py \
   --save_path ./ckpt/qwen3.5-9b-ppo \
   --ckpt_path ./ckpt/qwen3.5-9b-ppo-ckpt \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 8 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 8 \
   --n_samples_per_prompt "$PPO_N_SAMPLES_PER_PROMPT" \
   --num_episodes 2 \
   --max_epochs "$PPO_MAX_EPOCHS" \
   --max_samples 400000 \
   --max_len 4096 \
   --max_new_tokens "$PPO_MAX_NEW_TOKENS" \
   --zero_stage 3 \
   --param_dtype bf16 \
   --actor_learning_rate "$PPO_ACTOR_LR" \
   --critic_learning_rate 5e-6 \
   --lr_warmup_ratio 0.1 \
   --init_kl_coef "$PPO_INIT_KL_COEF" \
   --prompt_data /data/banlujie/data/bird_dev/bird_dev_ppo.jsonl \
   --input_key prompt \
   --label_key sql \
   --normalize_reward \
   --gradient_checkpointing \
   --vllm_sync_backend nccl \
   --train_max_tokens_per_gpu 4096 \
   --enable_vllm_is_correction \
   --attn_implementation flash_attention_2 \
   --use_wandb True \
   --adam_offload \
   --save_steps 20\
   --load_checkpoint \
   "${ASYNC_ARGS[@]}"
