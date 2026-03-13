import torch
import torch.nn as nn
from transformers import PreTrainedModel  # 假設 MiniBertConfig 繼承自 PretrainedConfig
from transformers.modeling_outputs import ModelOutput  # 使用更通用的輸出格式
from transformers import AutoTokenizer, AutoModelForMaskedLM,LongformerModel
from transformers import PretrainedConfig

AGGREGATE_MAP = {label: i for i, label in enumerate(["UNKNOWN", "NONE", "COUNT", "AVG", "SUM", "MAX", "MIN"])}
DIRECTION_MAP = {label: i for i, label in enumerate(["UNKNOWN", "NONE", "ASC", "DESC"])}


class SCM_Config(PretrainedConfig):
    """
    SCM 統一模型的設定類 (Configuration Class)。

    這個類別繼承自 `transformers.PretrainedConfig`，用於儲存 SCM 多任務模型的所有超參數。
    它不僅包含了基礎 Transformer Encoder 的名稱，還定義了所有特定任務預測頭的輸出維度。

    Attributes:
        base_model_name (str): 要使用的 Hugging Face 預訓練模型名稱，例如 "distilbert-base-uncased"。
        num_aggregates (int): 聚合函數預測頭的輸出維度 (類別數量)。
        num_directions (int): 排序方向預測頭的輸出維度 (例如: NONE, ASC, DESC)。
        num_quality_dims (int): 品質分數迴歸頭的輸出維度 (例如: 2, 代表 convergence 和 efficiency)。
    """
    model_type = "scm_model"

    def __init__(
        self,
        base_model_name: str = "allenai/longformer-base-4096",
        num_aggregates: int = 7,
        num_directions: int = 3,
        num_quality_dims: int = 2,
        **kwargs
    ):
        """
        初始化 SCM_Config 物件。

        Args:
            base_model_name (str, optional): 基礎預訓練模型的名稱。預設為 "distilbert-base-uncased"。
            num_aggregates (int, optional): 聚合函數的類別數量。預設為 7。
            num_directions (int, optional): 排序方向的類別數量。預設為 3。
            num_quality_dims (int, optional): 品質分數的維度數量。預設為 2。
            **kwargs: 傳遞給父類別 `PretrainedConfig` 的額外參數。
        """
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_aggregates = num_aggregates
        self.num_directions = num_directions
        self.num_quality_dims = num_quality_dims



class SCM_Unified_Model(PreTrainedModel):
    """
    最終的、統一的 SCM 模型。
    它結合了動態指針網絡來處理列，以及固定的頭來處理分類和迴歸任務。
    """
    config_class = SCM_Config

    def __init__(self, config: SCM_Config):
        super().__init__(config)
        self.config = config

        # 1. 共享的 Transformer Encoder Body
        # (這裡應放入您完整的 MiniBert Encoder 實現)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.encoder = AutoModelForMaskedLM.from_pretrained(
            config.base_model_name) if config.base_model_name != "allenai/longformer-base-4096" else LongformerModel.from_pretrained(config.base_model_name)

        # 例如: self.embeddings, self.encoder_layers, etc.
        # 2. 用於「指針/評分」機制的組件
        # 這個線性層用於將 CLS 向量轉換為一個統一的「查詢向量」
        encoder_hidden_size = self.encoder.config.hidden_size

        # 2. 動態指針/評分頭的組件 (保持不變)
        self.dynamic_head_transform = nn.Linear(encoder_hidden_size, encoder_hidden_size)

        # ======================================================
        # 3. 【完整實現】固定的分類頭
        # ======================================================
        self.aggregate_head = nn.Linear(encoder_hidden_size, config.num_aggregates)
        self.orderby_direction_head = nn.Linear(encoder_hidden_size, config.num_directions)

        # 4. 固定迴歸頭 (保持不變)
        self.quality_head = nn.Linear(encoder_hidden_size, config.num_quality_dims)  # for converge

    def forward(self,
                input_ids,
                attention_mask,
                column_token_indices,
                labels=None,
                **kwargs):
        # Encoder 前向傳播，得到所有 token 的最終隱層狀態
        # hidden_states 維度: (batch_size, seq_length, hidden_size)
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        # 獲取整個序列的表示 ([CLS] token)
        # cls_output 維度: (batch_size, hidden_size)
        cls_output = hidden_states[:, 0, :]

        # ======================================================
        # A. 處理「動態指針/評分」頭
        # ======================================================

        # 從 hidden_states 中根據索引提取所有列的表示向量
        # column_states 維度: (batch_size, num_columns_in_batch, hidden_size)
        batch_size = hidden_states.shape[0]
        col_indices_expanded = column_token_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        column_states = torch.gather(hidden_states, 1, col_indices_expanded)

        query_vec = self.dynamic_head_transform(cls_output)
        column_logits = torch.bmm(query_vec.unsqueeze(1), column_states.transpose(1, 2)).squeeze(1)


        # ======================================================
        # B. 處理「固定」頭
        # ======================================================

        aggregate_logits = self.aggregate_head(cls_output)
        orderby_dir_logits = self.orderby_direction_head(cls_output)

        # 預測品質分數 (輸出是一個數值)
        predicted_quality_scores = self.quality_head(cls_output)



        # ======================================================
        # C. 整合輸出
        # ======================================================

        # 返回一個包含所有預測的字典
        return {
            "column_logits": column_logits,
            "aggregate_logits": aggregate_logits,
            "orderby_direction_logits": orderby_dir_logits,
            "predicted_quality_scores": predicted_quality_scores
        }
