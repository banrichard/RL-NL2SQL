#!/bin/bash

# PPO训练脚本 - 本地Ray版本（避免上传大文件）
# 集成Credit Model进行细粒度的信用分配

# 设置基本路径
BASE_DIR="/home/banlujie/RL-NL2SQL"
ACTOR_MODEL_PATH="${BASE_DIR}/ckpt/phi4_mini_reasoning"
REWARD_MODEL_PATH="${BASE_DIR}/ckpt/phi4-7b-rm"
CREDIT_MODEL_PATH="${BASE_DIR}/ckpt/phi4-7b-cm"
SAVE_PATH="${BASE_DIR}/ckpt/phi4-7b-ppo-credit"
PROMPT_DATA_PATH="/home/banlujie/data/bird_train/train/sft/train_sft.json"
EVAL_DATASET="/home/banlujie/data/bird_train/train/sft/eval_sft.json"

# GPU配置
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
NUM_GPUS=${NUM_GPUS:-4}

# 检查所有必要的模型和数据是否存在
if [ ! -d "$CREDIT_MODEL_PATH" ]; then
    echo "Error: Credit Model not found at $CREDIT_MODEL_PATH"
    exit 1
fi

if [ ! -f "$PROMPT_DATA_PATH" ]; then
    echo "Warning: Prompt data not found at $PROMPT_DATA_PATH"
    echo "Using default OpenRLHF prompt collection instead"
    PROMPT_DATA_PATH="OpenRLHF/prompt-collection-v0.1"
    INPUT_KEY="context_messages"
    APPLY_CHAT_TEMPLATE="--apply_chat_template"
else
    INPUT_KEY="prompt"
    APPLY_CHAT_TEMPLATE="--apply_chat_template"
fi

echo "Starting PPO training with Credit Model integration using Ray (Local Mode)..."
echo "Actor Model: $ACTOR_MODEL_PATH"
echo "Reward Model: $REWARD_MODEL_PATH"
echo "Credit Model: $CREDIT_MODEL_PATH"
echo "Prompt Data: $PROMPT_DATA_PATH"
echo "Save Path: $SAVE_PATH"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# 设置代理服务器用于wandb
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

echo "Proxy settings for wandb:"
echo "HTTP_PROXY: $HTTP_PROXY"


cd "$BASE_DIR"

# 启动Ray集群
ray start --head --node-ip-address 0.0.0.0 --num-gpus $NUM_GPUS --num-cpus 8

# 等待Ray集群启动
sleep 5

# 直接运行Python脚本（不使用ray job submit）
python3 -m train_ppo_ray.py \
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
   --micro_train_batch_size 8 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 512 \
   --max_samples 50000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data $PROMPT_DATA_PATH \
   --input_key $INPUT_KEY \
   --eval_dataset $EVAL_DATASET \
   $APPLY_CHAT_TEMPLATE \
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
