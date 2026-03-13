#!/bin/bash

# SCM Decoder-Only 模型训练脚本
# 基于 Qwen2.5-0.5B + LoRA 微调
# 用于 PPO 训练中的 SQL 补全质量评估

set -e  # 出错立即退出

# ===================================================================
# 1. 打印配置信息
# ===================================================================
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        SCM Decoder-Only Model Training Script              ║"
echo "║        基础模型: Qwen3-0.6B + LoRA (r=16)                   ║"
echo "╚════════════════════════════════════════════════════════════╝"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "[1/5] 环境初始化..."
echo "      项目目录: $PROJECT_DIR"

# ===================================================================
# 2. 激活虚拟环境
# ===================================================================
VENV_PATH="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ 虚拟环境不存在: $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"
echo "✅ 虚拟环境已激活: $VENV_PATH"



# ===================================================================
# 4. 配置 CUDA 和模型缓存
# ===================================================================
echo "[3/5] 配置 CUDA 和缓存..."

# 使用指定的 GPU (可根据需要修改)
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-"1,2,3,5"}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 减少 HuggingFace 并行化（避免死锁）
export TOKENIZERS_PARALLELISM="false"
export NCCL_DEBUG="WARN"

# 模型缓存位置
export HF_HOME="${HF_HOME:-/data/banlujie/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

echo "TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo "HF_HOME=$HF_HOME"

# ===================================================================
# 6. 运行训练
# ===================================================================
echo "[5/5] 开始训练..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 trainer/SCM_trainer_decoder.py

# ===================================================================
# 7. 训练完成
# ===================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ 训练完成！"
echo ""
echo "输出位置:"
echo "  📁 模型: ./scm_decoder_trainer/final_checkpoint/"
echo "  📦 LoRA adapter: ./scm_decoder_trainer/final_checkpoint/lora_adapter/"
echo "  🔧 预测头权重: ./scm_decoder_trainer/final_checkpoint/head_weights.pt"
echo "  🔤 Tokenizer: ./scm_decoder_trainer/final_checkpoint/"
echo ""
echo "下一步:"
echo "  1. 在 PPO 训练中使用:"
echo "     from algorithm.scm_actor_decoder import SCM_DecoderModelActor"
echo "     --scm_pretrain ./scm_decoder_trainer/final_checkpoint"
echo "  2. 验证模型:"
echo "     python3 -c \"from algorithm.scm_processor_decoder import SCM_DecoderModel; m = SCM_DecoderModel.from_pretrained_with_lora('./scm_decoder_trainer/final_checkpoint'); print('✅ Model loaded')\""
echo ""
