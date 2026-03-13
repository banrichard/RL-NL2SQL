import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoTokenizer
import torch
import json
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from algorithm.scm_processor import SCM_Config, SCM_Unified_Model
from dataset.scm_dataset import SCM_AnalyzerDataset, scm_collate_fn

AGGREGATE_MAP = {label: i for i, label in enumerate(["UNKNOWN", "NONE", "COUNT", "AVG", "SUM", "MAX", "MIN"])}
DIRECTION_MAP = {label: i for i, label in enumerate(["UNKNOWN", "NONE", "ASC", "DESC"])}
SQL_INTENTS = {"SELECT", "FROM", "JOIN", "WHERE", "GROUP_BY", "HAVING", "ORDER_BY", "LIMIT", "AGGREGATE"}


class SCM_Trainer(Trainer):
    def __init__(self, quality_loss_weight=1.0, intent_loss_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 為了效率，將損失函數實例化一次
        self.loss_fn_quality = nn.MSELoss()
        self.loss_fn_clf_standard = nn.CrossEntropyLoss()
        # 這個版本專門用於 orderby_col，它會忽略 -100 的標籤
        self.loss_fn_clf_with_ignore = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_fn_multilabel = nn.BCEWithLogitsLoss()
        # 您可以在這裡加入損失權重等超參數
        self.quality_loss_weight = quality_loss_weight
        self.intent_loss_weight = intent_loss_weight

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        focused_intents_batch = labels.pop("focused_intents")

        # 2. 正常執行模型的前向傳播，獲得所有預測頭的 logits
        outputs = model(**inputs)
        device = next(model.parameters()).device  # 確保損失計算在正確的設備上
        batch_size = inputs["input_ids"].shape[0]
        # 初始化總損失張量，並確保它在正確的設備上
        total_loss = torch.tensor(0.0, device=device)

        # 3. 遍歷批次中的每一個樣本，動態計算其損失
        for i in range(batch_size):
            sample_loss = torch.tensor(0.0, device=device)
            focused_intents = set(focused_intents_batch[i])

            # --- A. 品質分數損失 (始終計算) ---
            # 每個思考步驟都有品質分數，所以這個損失總是被計算
            pred_scores = outputs["predicted_quality_scores"][i]
            true_scores = labels["quality"][i]
            sample_loss += self.quality_loss_weight * self.loss_fn_quality(pred_scores, true_scores)

            # --- B. SQL 意圖損失 (條件式計算) ---
            active_sql_intents = focused_intents.intersection(SQL_INTENTS)

            # 固定分類頭：Aggregate
            if "AGGREGATE" in active_sql_intents:
                # CrossEntropyLoss 期望 (N, C) 和 (N) 維度，我們用 unsqueeze 增加批次維度
                loss_agg = self.loss_fn_clf_standard(
                    outputs["aggregate_logits"][i].unsqueeze(0),
                    labels["aggregate"][i].unsqueeze(0)
                )
                sample_loss += self.intent_loss_weight * loss_agg

            # 動態指針/評分頭 (用於所有與欄位相關的意圖)
            # 這些意圖共享 column_logits 的輸出
            # 為簡化，我們假設如果任一相關意圖出現，就對所有相關標籤計算損失
            # 這種做法鼓勵模型學習一個通用的欄位評分表示
            column_intents = {"SELECT", "WHERE", "GROUP_BY"}
            if not active_sql_intents.isdisjoint(column_intents):
                # 對 SELECT 標籤計算多標籤損失
                if "SELECT" in active_sql_intents:
                    sample_loss += self.intent_loss_weight * self.loss_fn_multilabel(
                        outputs["column_logits"][i], labels["select"][i]
                    )
                # 對 GROUP_BY 標籤計算多標籤損失
                if "GROUP_BY" in active_sql_intents:
                    sample_loss += self.intent_loss_weight * self.loss_fn_multilabel(
                        outputs["column_logits"][i], labels["groupby"][i]
                    )
                # 對 WHERE 標籤計算多標籤損失
                if "WHERE" in active_sql_intents:
                    sample_loss += self.intent_loss_weight * self.loss_fn_multilabel(
                        outputs["column_logits"][i], labels["where"][i]
                    )

            # ORDER BY 意圖的特殊處理 (包含兩個頭)
            if "ORDER_BY" in focused_intents:
                # 1. 欄位選擇損失 (使用帶 ignore_index 的損失函數)
                loss_orderby_col = self.loss_fn_clf_with_ignore(
                    outputs["column_logits"][i].unsqueeze(0),
                    labels["orderby_col"][i].unsqueeze(0)
                )
                sample_loss += self.intent_loss_weight * loss_orderby_col

                # 2. 排序方向損失 (使用標準的損失函數)
                loss_orderby_dir = self.loss_fn_clf_standard(
                    outputs["orderby_direction_logits"][i].unsqueeze(0),
                    labels["orderby_dir"][i].unsqueeze(0)
                )
                sample_loss += self.intent_loss_weight * loss_orderby_dir
            # 可以在此處為 'FROM', 'JOIN' 等添加損失計算
            # 假設它們也使用某種 logits (例如 table_logits)

            total_loss += sample_loss

        # 4. 計算批次的平均損失
        avg_loss = total_loss / batch_size

        # 5. 返回 Trainer API 期望的格式
        return (avg_loss, outputs) if return_outputs else avg_loss


def main():
    """
    完整訓練流程的主執行函式。
    """
    # --- 基本設定 ---
    MODEL_NAME = "allenai/longformer-base-4096"
    TRAIN_DATA_FILE = "/data/banlujie/data/synsql_5k/train/enriched_scm_cot_all_full_train.jsonl"  # 您的輸入文件
    OUTPUT_DIR = "./scm_model_final_trainer1"

    # --- 訓練超參數 ---
    TRAINING_PARAMS = {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "fp16": torch.cuda.is_available(),
    }

    print("--- SCM 模型訓練 ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用設備: {device}")

    # 1. 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. 創建 Dataset
    # 確保您的 Dataset 類別和 collate_fn 是完整的
    # 此處為簡化，假設它們已定義好
    train_dataset = SCM_AnalyzerDataset(data_file_path=TRAIN_DATA_FILE, tokenizer=tokenizer,max_length=4096)
    print(f"訓練集加載完畢，共 {len(train_dataset)} 條樣本。")
    print("\n" + "=" * 50)
    print("DIAGNOSTIC REPORT")
    print(f"  - 數據文件路徑 (絕對路徑): {os.path.abspath(TRAIN_DATA_FILE)}")
    print(f"  - 數據集長度 (len(train_dataset)): {len(train_dataset)}")
    print(f"  - 批次大小 (per_device_train_batch_size): {TRAINING_PARAMS['per_device_train_batch_size']}")

    num_steps_per_epoch = len(train_dataset) // TRAINING_PARAMS['per_device_train_batch_size']
    total_steps = num_steps_per_epoch * TRAINING_PARAMS['num_train_epochs']
    print(f"  - 手動計算的總步數 (約): {total_steps}")
    print("=" * 50 + "\n")
    # 3. 創建模型 Config 和模型實例
    # 這裡的維度需要根據您的整體數據集來確定
    config = SCM_Config(
        base_model_name=MODEL_NAME,
        num_aggregates=len(AGGREGATE_MAP),
        num_directions=len(DIRECTION_MAP),
        num_quality_dims=2
    )
    model = SCM_Unified_Model(config)
    print("SCM 統一模型已成功創建。")

    # 4. 創建 TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        report_to="none",  # 避免需要 wandb 或 tensorboard
        **TRAINING_PARAMS
    )

    # 5. 實例化自訂的 SCM_Trainer
    trainer = SCM_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=scm_collate_fn,
        quality_loss_weight=1.0  # 您可以在這裡調整損失權重
    )

    # 6. 開始訓練
    print(f"\n🚀 開始使用 SCM_Trainer 進行微調... 訓練參數:")
    print(json.dumps(TRAINING_PARAMS, indent=2))

    trainer.train()

    print("\n✅ 訓練完成！")

    # 7. 保存最終模型
    final_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.save_model(final_checkpoint_dir)
    print(f"模型已保存至 {final_checkpoint_dir}")
    # tokenizer 也會被自動保存


if __name__ == "__main__":
    main()
