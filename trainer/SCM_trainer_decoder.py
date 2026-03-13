"""
SCM Trainer - Decoder-Only 版本
使用 HuggingFace Trainer API 训练 Decoder-Only SCM 模型。

与 Encoder-Only 版本的区别:
  1. 使用 SCM_DecoderModel (基于 Qwen2.5-0.5B + LoRA)
  2. 输入文本格式使用末尾 [COL] 标记
  3. 自动处理 [COL] 特殊标记的添加和 embedding resize
  4. 损失计算逻辑完全复用 (compute_loss 与原版一致)
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoTokenizer
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from algorithm.scm_processor_decoder import SCM_DecoderConfig, SCM_DecoderModel, COL_TOKEN
from dataset.scm_dataset_decoder import SCM_DecoderDataset, scm_decoder_collate_fn

AGGREGATE_MAP = {label: i for i, label in enumerate(["UNKNOWN", "NONE", "COUNT", "AVG", "SUM", "MAX", "MIN"])}
DIRECTION_MAP = {label: i for i, label in enumerate(["UNKNOWN", "NONE", "ASC", "DESC"])}
SQL_INTENTS = {"SELECT", "FROM", "JOIN", "WHERE", "GROUP_BY", "HAVING", "ORDER_BY", "LIMIT", "AGGREGATE"}


class SCM_DecoderTrainer(Trainer):
    """
    Decoder-Only SCM 训练器。

    损失计算逻辑与 Encoder-Only 版本完全一致:
      - quality_scores: MSE loss (始终计算)
      - aggregate: CrossEntropy (条件式)
      - column selection (select/groupby/where): BCEWithLogits (条件式)
      - orderby column: CrossEntropy with ignore_index (条件式)
      - orderby direction: CrossEntropy (条件式)
    """

    def __init__(self, quality_loss_weight=1.0, intent_loss_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn_quality = nn.MSELoss()
        self.loss_fn_clf_standard = nn.CrossEntropyLoss()
        self.loss_fn_clf_with_ignore = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_fn_multilabel = nn.BCEWithLogitsLoss(reduction='none')  # 逐元素, 手动 mask 后平均
        self.quality_loss_weight = quality_loss_weight
        self.intent_loss_weight = intent_loss_weight

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        focused_intents_batch = labels.pop("focused_intents")
        # 列 mask: 1=真实列, 0=padding列 (用于屏蔽 padding 列的 loss)
        column_mask = inputs.pop("column_mask").to(next(model.parameters()).device) if "column_mask" in inputs else None

        # 模型前向传播
        outputs = model(**inputs)
        device = next(model.parameters()).device
        batch_size = inputs["input_ids"].shape[0]
        total_loss = torch.tensor(0.0, device=device)

        # 将模型输出统一转为 float32，避免 bf16 混合精度下 backward dtype 不匹配
        select_logits = outputs["select_logits"].float()
        where_logits = outputs["where_logits"].float()
        groupby_logits = outputs["groupby_logits"].float()
        orderby_col_logits = outputs["orderby_col_logits"].float()
        aggregate_logits = outputs["aggregate_logits"].float()
        orderby_dir_logits = outputs["orderby_direction_logits"].float()
        quality_scores = outputs["predicted_quality_scores"].float()

        for i in range(batch_size):
            sample_loss = torch.tensor(0.0, device=device)
            focused_intents = set(focused_intents_batch[i])
            num_loss_terms = 0  # 统计激活的 loss 项数，用于归一化

            # --- A. 品质分数损失 (始终计算) ---
            # quality labels 范围 [0,10], 归一化到 [0,1] 与 sigmoid 输出匹配
            pred_scores = quality_scores[i]
            true_scores = labels["quality"][i].to(device) / 10.0
            sample_loss += self.quality_loss_weight * self.loss_fn_quality(pred_scores, true_scores)
            num_loss_terms += 1

            # --- B. SQL 意图损失 (条件式计算) ---
            active_sql_intents = focused_intents.intersection(SQL_INTENTS)

            # Aggregate 分类
            if "AGGREGATE" in active_sql_intents:
                loss_agg = self.loss_fn_clf_standard(
                    aggregate_logits[i].unsqueeze(0),
                    labels["aggregate"][i].unsqueeze(0).to(device)
                )
                sample_loss += self.intent_loss_weight * loss_agg
                num_loss_terms += 1

            # 列选择 — 每个任务使用独立的 logits, 用 column_mask 屏蔽 padding 列
            mask_i = column_mask[i] if column_mask is not None else torch.ones_like(select_logits[i])

            if "SELECT" in active_sql_intents:
                raw_loss = self.loss_fn_multilabel(select_logits[i], labels["select"][i].to(device))
                sample_loss += self.intent_loss_weight * (raw_loss * mask_i).sum() / mask_i.sum().clamp(min=1)
                num_loss_terms += 1
            if "GROUP_BY" in active_sql_intents:
                raw_loss = self.loss_fn_multilabel(groupby_logits[i], labels["groupby"][i].to(device))
                sample_loss += self.intent_loss_weight * (raw_loss * mask_i).sum() / mask_i.sum().clamp(min=1)
                num_loss_terms += 1
            if "WHERE" in active_sql_intents:
                raw_loss = self.loss_fn_multilabel(where_logits[i], labels["where"][i].to(device))
                sample_loss += self.intent_loss_weight * (raw_loss * mask_i).sum() / mask_i.sum().clamp(min=1)
                num_loss_terms += 1

            # ORDER BY — 使用独立的 orderby_col_logits, 只在真实列范围内计算 CE
            if "ORDER_BY" in focused_intents:
                num_real_cols = int(mask_i.sum().item())
                orderby_target = labels["orderby_col"][i].unsqueeze(0).to(device)
                loss_orderby_col = self.loss_fn_clf_with_ignore(
                    orderby_col_logits[i, :num_real_cols].unsqueeze(0),
                    orderby_target
                )
                # 当所有 target 都是 ignore_index 时, CE 返回 nan, 替换为 0
                if not torch.isnan(loss_orderby_col):
                    sample_loss += self.intent_loss_weight * loss_orderby_col
                    num_loss_terms += 1

                loss_orderby_dir = self.loss_fn_clf_standard(
                    orderby_dir_logits[i].unsqueeze(0),
                    labels["orderby_dir"][i].unsqueeze(0).to(device)
                )
                sample_loss += self.intent_loss_weight * loss_orderby_dir
                num_loss_terms += 2

            # 归一化: 除以激活的 loss 项数，避免 intent 数量不同导致 loss 尺度差异
            total_loss += sample_loss / max(num_loss_terms, 1)

        avg_loss = total_loss / batch_size
        return (avg_loss, outputs) if return_outputs else avg_loss


def main():
    """
    Decoder-Only SCM 模型完整训练流程。
    """
    # --- 配置 ---
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    TRAIN_DATA_FILE = "/data/banlujie/data/synsql_5k/train/enriched_scm_cot_all_full_train.jsonl"
    OUTPUT_DIR = "./scm_decoder_trainer"

    TRAINING_PARAMS = {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,      # decoder 模型更大，batch 适当减小
        "gradient_accumulation_steps": 4,       # 等效 batch_size = 16
        "learning_rate": 2e-4,                  # LoRA 适合较高学习率
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "bf16": torch.cuda.is_available(),
        "gradient_checkpointing": False,         # 节省显存
        "dataloader_pin_memory": True,      
        "max_grad_norm": 1.0,                       # 梯度裁剪，稳定训练   
        }

    # LoRA 配置
    LORA_CONFIG = {
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
    }

    print("=" * 60)
    print("  SCM Decoder-Only 模型训练")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"基础模型: {MODEL_NAME}")
    print(f"LoRA: r={LORA_CONFIG['lora_r']}, alpha={LORA_CONFIG['lora_alpha']}")

    # 1. 初始化 Tokenizer + 添加 [COL] 特殊标记
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": [COL_TOKEN]})
    print(f"添加了 {num_added} 个特殊标记: {COL_TOKEN} -> id={tokenizer.convert_tokens_to_ids(COL_TOKEN)}")

    # 2. 创建 Dataset
    MAX_LENGTH = 2048  # Decoder 模型一般不需要 4096
    train_dataset = SCM_DecoderDataset(
        data_file_path=TRAIN_DATA_FILE,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    print(f"训练集加载完毕，共 {len(train_dataset)} 条样本。")

    # 3. 创建模型
    config = SCM_DecoderConfig(
        base_model_name=MODEL_NAME,
        num_aggregates=len(AGGREGATE_MAP),
        num_directions=len(DIRECTION_MAP),
        num_quality_dims=2,
        **LORA_CONFIG
    )
    model = SCM_DecoderModel(config)

    # 调整 embedding 大小以适应新增的 [COL] 标记
    if num_added > 0:
        # 获取底层模型 (LoRA 包装下需要 unwrap)
        base_model = model.decoder
        if hasattr(base_model, 'base_model'):
            # PEFT 包装
            base_model.resize_token_embeddings(len(tokenizer))
        else:
            base_model.resize_token_embeddings(len(tokenizer))
        print(f"Token embeddings 已调整为 {len(tokenizer)} 个")

    print("SCM Decoder-Only 模型已成功创建。")

    # 诊断信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- 模型参数统计 ---")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  可训练比例: {trainable_params / total_params * 100:.2f}%")

    num_steps_per_epoch = len(train_dataset) // (
        TRAINING_PARAMS['per_device_train_batch_size'] * TRAINING_PARAMS['gradient_accumulation_steps']
    )
    total_steps = num_steps_per_epoch * TRAINING_PARAMS['num_train_epochs']
    print(f"  预计总训练步数: ~{total_steps}")

    # 4. TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        report_to="none",
        remove_unused_columns=False,  # 自定义 collate_fn 需要这个
        **TRAINING_PARAMS
    )

    # 5. Trainer
    trainer = SCM_DecoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=scm_decoder_collate_fn,
        quality_loss_weight=1.0,
        intent_loss_weight=1.0,
    )

    # 6. 训练
    print(f"\n🚀 开始训练... 参数:")
    print(json.dumps(TRAINING_PARAMS, indent=2))
    trainer.train()
    print("\n✅ 训练完成！")

    # 7. 保存 (使用 LoRA 专用保存方法)
    final_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    model.save_pretrained_with_lora(final_checkpoint_dir)
    # 保存 tokenizer (包含 [COL] 标记)
    tokenizer.save_pretrained(final_checkpoint_dir)
    print(f"模型已保存至 {final_checkpoint_dir}")
    print(f"  - LoRA adapter: {final_checkpoint_dir}/lora_adapter/")
    print(f"  - 预测头权重: {final_checkpoint_dir}/head_weights.pt")
    print(f"  - Tokenizer: {final_checkpoint_dir}/")

    # 可选: 验证加载
    print("\n🔍 验证模型加载...")
    loaded_model = SCM_DecoderModel.from_pretrained_with_lora(final_checkpoint_dir)
    print("✅ 模型加载验证通过！")


if __name__ == "__main__":
    main()
