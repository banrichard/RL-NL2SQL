#!/bin/bash

# 设置基本路径
BASE_DIR="/home/banlujie/RL-NL2SQL"
ACTOR_MODEL_PATH="${BASE_DIR}/ckpt/phi4_mini_reasoning"
REWARD_MODEL_PATH="${BASE_DIR}/ckpt/phi4-7b-rm"
CREDIT_MODEL_PATH="${BASE_DIR}/ckpt/phi4-7b-cm"
SAVE_PATH="${BASE_DIR}/ckpt/phi4-7b-ppo-credit"
# 自定义你的prompt数据路径
PROMPT_DATA_PATH="/home/banlujie/data/bird_train/train/sft/train_sft.json"
EVAL_DATASET="/home/banlujie/data/bird_train/train/sft/eval_sft.json"
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
    INPUT_KEY="prompt"  # 根据你的数据格式调整
    APPLY_CHAT_TEMPLATE="--apply_chat_template"
fi

# GPU配置 - 可以修改这里来指定使用哪些GPU
# 例如: "0,1,2,3" 使用前4块GPU, "2,3,4,5" 使用GPU 2-5
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,4,6"}
NUM_GPUS=${NUM_GPUS:-4}

echo "Starting PPO training with Credit Model integration using Ray..."
echo "Actor Model: $ACTOR_MODEL_PATH"
echo "Reward Model: $REWARD_MODEL_PATH"
echo "Credit Model: $CREDIT_MODEL_PATH"
echo "Prompt Data: $PROMPT_DATA_PATH"
echo "Save Path: $SAVE_PATH"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# 启动Ray集群，使用指定的GPU
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# 设置代理服务器用于wandb
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890


echo "Proxy settings:"
echo "HTTP_PROXY: $http_proxy"
echo "HTTPS_PROXY: $https_proxy"

ray start --head --node-ip-address 0.0.0.0 --num-gpus $NUM_GPUS --num-cpus 16

# 创建运行时环境配置，排除大文件并包含代理设置
RUNTIME_ENV='{
  "working_dir": "/home/banlujie/RL-NL2SQL",
  "env_vars": {
    "http_proxy": "http://127.0.0.1:7890",
    "https_proxy": "http://127.0.0.1:7890"
  },
  "excludes": [
    ".git/**",
    "wandb/**", 
    "ckpt/**/*.safetensors",
    "ckpt/**/tokenizer.json",
    "*.log",
    "**/*.wandb",
    "__pycache__/**",
    "*.pyc",
    ".venv/**"
  ]
}'

# python -m openrlhf.cli.serve_rm \
#     --reward_pretrain $REWARD_MODEL_PATH \
#     --port 5000 \
#     --bf16 \
#     --attn_implementation flash_attention_2 \
#     --normalize_reward \
#     --max_len 8192 \
#     --batch_size 16

# 使用Ray进行PPO训练，集成Credit Model
# ray job submit --address="http://127.0.0.1:8265" \
#    --runtime-env-json="$RUNTIME_ENV" \
python3 train_ppo_ray.py \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --credit_num_nodes 1 \
   --credit_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.9 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain $ACTOR_MODEL_PATH \
   --credit_pretrain $CREDIT_MODEL_PATH \
   --reward_pretrain $REWARD_MODEL_PATH \
   --save_path $SAVE_PATH \
   --micro_train_batch_size 4 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 256 \
  --num_episodes 3 \
  --max_samples 200000 \
  --max_epochs 4 \
   --max_len 2048 \
   --zero_stage 3 \
   --bf16 \
  --actor_learning_rate 2e-6 \
  --critic_learning_rate 5e-6 \
  --lr_warmup_ratio 0.1 \
  --init_kl_coef 0.0005 \
   --prompt_data $PROMPT_DATA_PATH \
   --input_key $INPUT_KEY \
   --label_key SQL \
   $APPLY_CHAT_TEMPLATE \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --attn_implementation flash_attention_2 \
   --gradient_checkpointing \
   --save_steps 100 \
   --logging_steps 10 \
   --eval_steps 100 \
   --use_wandb 5c00be123cfc09b2a58a5d1071ffbfba53cd88d3 \
   --wandb_project "RL-NL2SQL-PPO-Credit" \
   --wandb_run_name "ppo_with_credit_$(date +%m%d_%H%M)" \
   --credit_num_outputs 8 \
   --credit_weight 0.1 \
   --enable_credit_shaping \
   --credit_aggregation "weighted_sum" \
   --enable_separated_rewards \
   --clause_credit_weight 0.2 \
   --clause_discount_factor 0.99 \
   --enable_adaptive_entropy \
   --entropy_threshold 2.0 \
   --low_k 5 \
   --high_k 50 \
   --enable_content_filtering \
   --use_dynamic_k \
   --k_min 5 \
   --k_max 50 \
   --sigmoid_weight 2.0 \
   --entropy_center 3.0 \
   --vllm_sync_backend nccl \
   --use_dynamic_batch \
   --rollout_max_tokens_per_gpu 8192 \
   --train_max_tokens_per_gpu 8192

echo "PPO training with Credit Model completed!"

ray stop
