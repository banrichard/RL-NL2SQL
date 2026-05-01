export RAY_local_fs_capacity_threshold=0.99
export RAY_memory_usage_threshold=0.99
export REWARD_MODEL_PATH=/home/banlujie/RL-NL2SQL/ckpt/qwen3-0.6b-rm
# REWARD_FUNC_AGGREGATION is for offline preference-data construction, not online reward.
export REMOTE_RM_URL_FOR_AGENT=/data/banlujie/RL-NL2SQL/algorithm/reward_func_token_level.py
# Reward-model device inside the process. With CUDA_VISIBLE_DEVICES=2,7:
#   cuda:0 -> physical GPU 2, cuda:1 -> physical GPU 7
export REWARD_MODEL_DEVICE="${REWARD_MODEL_DEVICE:-cuda:0}"
# Reward model memory is controlled independently from --vllm_gpu_memory_utilization.
# Either set an absolute cap (e.g. 8GiB), or a fraction (e.g. 0.12).
export REWARD_MODEL_MAX_MEMORY="${REWARD_MODEL_MAX_MEMORY:-}"
export REWARD_MODEL_GPU_MEMORY_UTILIZATION="${REWARD_MODEL_GPU_MEMORY_UTILIZATION:-0.12}"
export VLLM_THINKING_TOKEN_BUDGET="${VLLM_THINKING_TOKEN_BUDGET:-512}"
export VLLM_ANSWER_TOKEN_RESERVE="${VLLM_ANSWER_TOKEN_RESERVE:-1024}"
export VLLM_STOP_SEQUENCES="${VLLM_STOP_SEQUENCES:-</answer>,<|im_end|>,</s>}"
export VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
export STOP_PROPERLY_PENALTY_COEF="${STOP_PROPERLY_PENALTY_COEF:-0.0}"
export GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-0.4}"
export GRPO_INIT_KL_COEF="${GRPO_INIT_KL_COEF:-0.01}"
export REWARD_ENABLE_EXEC="${REWARD_ENABLE_EXEC:-1}"
export REWARD_EXEC_PARSE_FAIL="${REWARD_EXEC_PARSE_FAIL:--1.0}"
export REWARD_EXEC_MATCH_REWARD="${REWARD_EXEC_MATCH_REWARD:-1.0}"
export REWARD_EXEC_TIMEOUT_REWARD="${REWARD_EXEC_TIMEOUT_REWARD:--0.6}"
export REWARD_EXEC_ERROR_REWARD="${REWARD_EXEC_ERROR_REWARD:--0.4}"
export REWARD_EXEC_TIMEOUT_SEC="${REWARD_EXEC_TIMEOUT_SEC:-5.0}"
export REWARD_FALLBACK_UNCLOSED_ANSWER="${REWARD_FALLBACK_UNCLOSED_ANSWER:-0.0}"
export REWARD_FALLBACK_MISSING_ANSWER="${REWARD_FALLBACK_MISSING_ANSWER:-0.0}"
export REWARD_FALLBACK_THINK_OVER_BUDGET_PENALTY="${REWARD_FALLBACK_THINK_OVER_BUDGET_PENALTY:-0.0}"
export REWARD_FALLBACK_THINK_OVER_BUDGET_NO_ANSWER_PENALTY="${REWARD_FALLBACK_THINK_OVER_BUDGET_NO_ANSWER_PENALTY:-0.0}"
export REWARD_BIRD_TABLES_FILES="${REWARD_BIRD_TABLES_FILES:-/data/banlujie/data/bird_dev/dev_tables.json,/data/banlujie/data/bird_train/train/train_tables.json}"
export REWARD_BIRD_DB_ROOTS="${REWARD_BIRD_DB_ROOTS:-/data/banlujie/data/bird_dev/dev_databases,/data/banlujie/data/bird_train/train/train_databases}"

# GRPO uses group_norm advantage estimator — no critic model needed.
# This frees one GPU vs PPO. With colocate enabled: actor+ref+vllm share GPUs efficiently.
# Use GPUs 1,2 only (GPU 6,7 have hardware faults). Colocate allows 1 vllm engine with 2 actor/ref.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

# GRPO key differences from PPO:
#   - No critic (--advantage_estimator group_norm disables critic automatically)
#   - KL controlled via --use_kl_loss + --kl_estimator k3 (in-loss, not in-reward)
#   - Larger n_samples_per_prompt (group size) for better advantage estimation
#   - CRITICAL: --colocate_all_models to share GPU workers (avoid OOM)
FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --vllm_gpu_memory_utilization 0.45 \
   --vllm_reasoning_parser "$VLLM_REASONING_PARSER" \
   --gamma 1.0 \
   --temperature "$GRPO_TEMPERATURE" \
   --advantage_estimator group_norm \
   --use_kl_loss \
   --kl_estimator k3 \
   --init_kl_coef "$GRPO_INIT_KL_COEF" \
   --pretrain /home/banlujie/RL-NL2SQL/ckpt/Qwen3.5-9B \
   --remote_rm_url /data/banlujie/RL-NL2SQL/algorithm/reward_func_token_level.py\
   --agent_func_path /data/banlujie/RL-NL2SQL/algorithm/agent_executor_with_thinking_budget.py \
   --save_path ./ckpt/qwen3.5-9b-ppo-ablation-token \
   --ckpt_path ./ckpt/qwen3.5-9b-grpo-ckpt-ablation \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 16 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 16 \
   --n_samples_per_prompt 4 \
   --num_episodes 2 \
   --max_epochs 1 \
   --max_samples 400000 \
   --max_len 4096\
   --max_new_tokens 1536 \
   --stop_properly_penalty_coef "$STOP_PROPERLY_PENALTY_COEF" \
   --zero_stage 3 \
   --param_dtype bf16 \
   --actor_learning_rate 2e-6 \
   --lr_warmup_ratio 0.1 \
   --prompt_data /data/banlujie/data/bird_dev/bird_dev_ppo.jsonl \
   --input_key prompt \
   --label_key sql \
   --gradient_checkpointing \
   --vllm_sync_backend nccl \
   --enable_vllm_is_correction \
   --attn_implementation flash_attention_2 \
   --use_wandb True \
   --adam_offload \
   --train_max_tokens_per_gpu 8192 \
   --colocate_all_models \
   --save_steps 20 \
   #--load_checkpoint \
   # Removed --packing_samples and --use_dynamic_batch: unstable with flash_attn 2.8.x
   #--eval_dataset /data/banlujie/data/bird_dev/bird_dev_ppo.jsonl \
   # Removed --normalize_reward: GRPO's group_norm already normalizes advantages
