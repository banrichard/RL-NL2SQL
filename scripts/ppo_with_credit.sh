#!/bin/bash

# PPO训练脚本，集成Credit Model进行细粒度的信用分配
# 使用你已经训练好的Credit Model来增强PPO训练
# 使用Ray进行分布式训练

# 设置基本路径
BASE_DIR="/home/banlujie/RL-NL2SQL"
ACTOR_MODEL_PATH="${BASE_DIR}/ckpt/phi4_mini_reasoning"  # 你的SFT模型
REWARD_MODEL_PATH="${BASE_DIR}/ckpt/phi4-7b-rm"         # 你的奖励模型
CREDIT_MODEL_PATH="${BASE_DIR}/ckpt/phi4-7b-cm"         # 你训练好的Credit Model
SAVE_PATH="${BASE_DIR}/ckpt/phi4-7b-ppo-credit"        # PPO训练输出路径

# 检查Credit Model是否存在
if [ ! -d "$CREDIT_MODEL_PATH" ]; then
    echo "Error: Credit Model not found at $CREDIT_MODEL_PATH"
    echo "Please make sure you have trained the Credit Model first using train_cm.py"
    exit 1
fi

echo "Starting PPO training with Credit Model integration using Ray..."
echo "Actor Model: $ACTOR_MODEL_PATH"
echo "Reward Model: $REWARD_MODEL_PATH"
echo "Credit Model: $CREDIT_MODEL_PATH"
echo "Save Path: $SAVE_PATH"

# 使用Ray进行PPO训练，集成Credit Model
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/home/banlujie/RL-NL2SQL"}' \
   -- python3 -m train_ppo_ray.py \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --credit_num_nodes 1 \
   --credit_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain $ACTOR_MODEL_PATH \
   --reward_pretrain $REWARD_MODEL_PATH \
   --credit_pretrain $CREDIT_MODEL_PATH \
   --save_path $SAVE_PATH \
   --micro_train_batch_size 16 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --attn_implementation flash_attention_2 \
   --gradient_checkpointing \
   --save_steps 500 \
   --logging_steps 10 \
   --eval_steps 500 \
   --use_wandb 5c00be123cfc09b2a58a5d1071ffbfba53cd88d3 \
   --wandb_project "RL-NL2SQL-PPO-Credit" \
   --wandb_run_name "ppo_with_credit_$(date +%m%d_%H%M)" \
   --credit_num_outputs 8 \
   --credit_weight 0.1 \
   --enable_credit_shaping \
   --credit_aggregation "weighted_sum"

echo "PPO training with Credit Model completed!"
