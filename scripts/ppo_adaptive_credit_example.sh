#!/bin/bash

# PPO训练脚本 - 集成AdaptiveEntropyProcessor和子句级别Credit Model
# 支持动态Top-K采样和实时信用分配
# 使用Ray进行分布式训练

# 设置基本路径
BASE_DIR="/home/banlujie/RL-NL2SQL"
ACTOR_MODEL_PATH="${BASE_DIR}/ckpt/phi4_mini_reasoning"
REWARD_MODEL_PATH="${BASE_DIR}/ckpt/phi4-7b-rm"
CREDIT_MODEL_PATH="${BASE_DIR}/ckpt/phi4-7b-cm"
SAVE_PATH="${BASE_DIR}/ckpt/phi4-7b-ppo-adaptive-credit"

# 自定义数据路径
PROMPT_DATA_PATH="/home/banlujie/data/bird_train/train/sft/train_sft.json"
EVAL_DATASET="/home/banlujie/data/bird_train/train/sft/eval_sft.json"

# 检查所有必要的模型和数据是否存在
echo "=== 检查模型和数据文件 ==="
if [ ! -d "$ACTOR_MODEL_PATH" ]; then
    echo "Error: Actor Model not found at $ACTOR_MODEL_PATH"
    exit 1
fi

if [ ! -d "$REWARD_MODEL_PATH" ]; then
    echo "Error: Reward Model not found at $REWARD_MODEL_PATH"
    exit 1
fi

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

# GPU配置
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5"}
NUM_GPUS=${NUM_GPUS:-6}

# AdaptiveEntropyProcessor配置
ENABLE_ADAPTIVE_ENTROPY=${ENABLE_ADAPTIVE_ENTROPY:-true}
LOW_ENTROPY_THRESHOLD=${LOW_ENTROPY_THRESHOLD:-2.0}
HIGH_ENTROPY_THRESHOLD=${HIGH_ENTROPY_THRESHOLD:-4.0}
LOW_K=${LOW_K:-5}
HIGH_K=${HIGH_K:-50}
ENABLE_CONTENT_FILTERING=${ENABLE_CONTENT_FILTERING:-true}

# 子句级别Credit Model配置
ENABLE_CLAUSE_CREDIT=${ENABLE_CLAUSE_CREDIT:-true}
CLAUSE_DISCOUNT_FACTOR=${CLAUSE_DISCOUNT_FACTOR:-0.99}
CLAUSE_CREDIT_WEIGHT=${CLAUSE_CREDIT_WEIGHT:-0.2}

echo "=== PPO训练配置 ==="
echo "Actor Model: $ACTOR_MODEL_PATH"
echo "Reward Model: $REWARD_MODEL_PATH"
echo "Credit Model: $CREDIT_MODEL_PATH"
echo "Prompt Data: $PROMPT_DATA_PATH"
echo "Save Path: $SAVE_PATH"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

echo ""
echo "=== AdaptiveEntropyProcessor配置 ==="
echo "启用自适应熵处理器: $ENABLE_ADAPTIVE_ENTROPY"
echo "低熵阈值: $LOW_ENTROPY_THRESHOLD (使用k=$LOW_K进行exploitation)"
echo "高熵阈值: $HIGH_ENTROPY_THRESHOLD (使用k=$HIGH_K进行exploration)"
echo "内容过滤: $ENABLE_CONTENT_FILTERING"

echo ""
echo "=== 子句级别Credit Model配置 ==="
echo "启用子句级别信用分配: $ENABLE_CLAUSE_CREDIT"
echo "折扣因子: $CLAUSE_DISCOUNT_FACTOR"
echo "子句信用权重: $CLAUSE_CREDIT_WEIGHT"

# 设置代理服务器
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

echo ""
echo "=== 启动Ray集群 ==="
ray start --head --node-ip-address 0.0.0.0 --num-gpus $NUM_GPUS --num-cpus 16

# 构建AdaptiveEntropyProcessor参数
ADAPTIVE_ENTROPY_ARGS=""
if [ "$ENABLE_ADAPTIVE_ENTROPY" = "true" ]; then
    ADAPTIVE_ENTROPY_ARGS="--enable_adaptive_entropy \
                          --low_entropy_threshold $LOW_ENTROPY_THRESHOLD \
                          --high_entropy_threshold $HIGH_ENTROPY_THRESHOLD \
                          --low_k $LOW_K \
                          --high_k $HIGH_K"
    
    if [ "$ENABLE_CONTENT_FILTERING" = "true" ]; then
        ADAPTIVE_ENTROPY_ARGS="$ADAPTIVE_ENTROPY_ARGS --enable_content_filtering"
    fi
fi

# 构建子句级别Credit Model参数
CLAUSE_CREDIT_ARGS=""
if [ "$ENABLE_CLAUSE_CREDIT" = "true" ]; then
    CLAUSE_CREDIT_ARGS="--enable_clause_credit \
                       --clause_discount_factor $CLAUSE_DISCOUNT_FACTOR \
                       --clause_credit_weight $CLAUSE_CREDIT_WEIGHT"
fi

echo ""
echo "=== 开始PPO训练 ==="
echo "集成功能："
echo "  ✓ 基础PPO算法"
echo "  ✓ Credit Model细粒度信用分配"
if [ "$ENABLE_ADAPTIVE_ENTROPY" = "true" ]; then
    echo "  ✓ AdaptiveEntropyProcessor动态采样"
fi
if [ "$ENABLE_CLAUSE_CREDIT" = "true" ]; then
    echo "  ✓ 子句级别实时信用分配"
fi

echo ""
echo "训练命令："
echo "python3 train_ppo_ray.py \\"

# 执行PPO训练
python3 train_ppo_ray.py \
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
   --colocate_critic_reward \
   --colocate_actor_ref \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.6 \
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
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data $PROMPT_DATA_PATH \
   --input_key $INPUT_KEY \
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
   --wandb_project "RL-NL2SQL-PPO-AdaptiveCredit" \
   --wandb_run_name "ppo_adaptive_credit_$(date +%m%d_%H%M)" \
   --credit_num_outputs 8 \
   --credit_weight 0.1 \
   --enable_credit_shaping \
   --credit_aggregation "weighted_sum" \
   --vllm_sync_backend nccl \
   --use_dynamic_batch \
   --rollout_max_tokens_per_gpu 16384 \
   --train_max_tokens_per_gpu 16384 \
   $ADAPTIVE_ENTROPY_ARGS \
   $CLAUSE_CREDIT_ARGS

TRAINING_EXIT_CODE=$?

echo ""
echo "=== 训练结果 ==="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ PPO训练成功完成！"
    echo ""
    echo "训练总结："
    echo "- 模型保存路径: $SAVE_PATH"
    echo "- WandB项目: RL-NL2SQL-PPO-AdaptiveCredit"
    echo "- 集成功能: PPO + Credit Model + AdaptiveEntropy + 子句级别信用分配"
    
    if [ "$ENABLE_ADAPTIVE_ENTROPY" = "true" ]; then
        echo ""
        echo "AdaptiveEntropyProcessor效果："
        echo "- 低熵场景: 使用Top-K=$LOW_K进行精确生成"
        echo "- 高熵场景: 使用Top-K=$HIGH_K进行探索性生成"
        echo "- 动态平衡: exploration ↔ exploitation"
    fi
    
    if [ "$ENABLE_CLAUSE_CREDIT" = "true" ]; then
        echo ""
        echo "子句级别信用分配效果："
        echo "- 实时计算每个子句的信用分"
        echo "- 折扣归因分配到每个Token"
        echo "- 增强SQL结构学习"
    fi
    
    echo ""
    echo "使用训练好的模型："
    echo "export MODEL_PATH=\"$SAVE_PATH\""
    echo "python generate_sql.py --model \$MODEL_PATH --input \"your_question\""
    
else
    echo "❌ PPO训练失败，退出码: $TRAINING_EXIT_CODE"
    echo "请检查日志以获取详细错误信息"
fi

# 清理Ray集群
echo ""
echo "=== 清理资源 ==="
ray stop
echo "Ray集群已停止"

exit $TRAINING_EXIT_CODE
