"""
SCM Processor - Decoder-Only 版本
使用因果语言模型 (如 Qwen2.5-0.5B) 作为 backbone，配合 LoRA 微调。
通过在序列末尾追加 [COL] 标记来解决因果注意力下的列表示问题。

核心设计：
  输入格式: question + schema_ddl + cot_history + [COL]col1[COL]col2...
  序列表示: 取最后一个非 padding token 的 hidden state (替代 [CLS])
  列表示:   从 [COL] 标记位置提取 hidden states
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from transformers import PretrainedConfig
from peft import get_peft_model, PeftModel, LoraConfig, TaskType

AGGREGATE_MAP = {label: i for i, label in enumerate(["UNKNOWN", "NONE", "COUNT", "AVG", "SUM", "MAX", "MIN"])}
DIRECTION_MAP = {label: i for i, label in enumerate(["UNKNOWN", "NONE", "ASC", "DESC"])}

# 特殊标记，用于标识列位置
COL_TOKEN = "[COL]"


class SCM_DecoderConfig(PretrainedConfig):
    """
    Decoder-Only SCM 模型的配置类。

    Attributes:
        base_model_name (str): Decoder-only 预训练模型名称。
        num_aggregates (int): 聚合函数预测头的输出维度。
        num_directions (int): 排序方向预测头的输出维度。
        num_quality_dims (int): 品质分数回归头的输出维度。
        use_lora (bool): 是否使用 LoRA 微调。
        lora_r (int): LoRA rank。
        lora_alpha (int): LoRA alpha。
        lora_dropout (float): LoRA dropout。
        lora_target_modules (list): LoRA 目标模块列表。
    """
    model_type = "scm_decoder_model"

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-0.6B",
        num_aggregates: int = 7,
        num_directions: int = 4,
        num_quality_dims: int = 2,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_aggregates = num_aggregates
        self.num_directions = num_directions
        self.num_quality_dims = num_quality_dims
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]


class SCM_DecoderModel(PreTrainedModel):
    """
    Decoder-Only SCM 统一模型。

    架构:
      1. Decoder-Only LLM backbone (可选 LoRA)
      2. 每任务独立的动态指针头: select/where/groupby/orderby_col, 各自从 [COL] 位置提取列表示做 scaled dot-product
      3. 固定分类头: aggregate, orderby_direction
      4. 回归头: quality_scores

    输入格式:
      question\nschema_ddl\ncot_history\n[COL]table.col1[COL]table.col2...
    """
    config_class = SCM_DecoderConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: SCM_DecoderConfig):
        super().__init__(config)
        self.config = config

        # 1. 加载 Decoder-Only backbone
        self.decoder = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            dtype=torch.bfloat16,
            output_hidden_states=True,
        )

        # 2. 可选: 应用 LoRA
        if config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
                inference_mode=False,
            )
            # gradient_checkpointing 需要 input 有 requires_grad
            self.decoder.enable_input_require_grads()
            self.decoder = get_peft_model(self.decoder, lora_config)
            self.decoder.print_trainable_parameters()

        # 获取 hidden_size
        decoder_config = self.decoder.config if not config.use_lora else self.decoder.base_model.model.config
        hidden_size = decoder_config.hidden_size

        # 3. 每个列选择任务独立的 query 变换头 (避免共用 logits 导致梯度冲突)
        self.select_head = nn.Linear(hidden_size, hidden_size)
        self.where_head = nn.Linear(hidden_size, hidden_size)
        self.groupby_head = nn.Linear(hidden_size, hidden_size)
        self.orderby_col_head = nn.Linear(hidden_size, hidden_size)

        # LayerNorm: 在 dot-product 前归一化 query 和 key, 控制 logits 幅度
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)

        # 4. 固定分类头 (也需要归一化输入)
        self.seq_norm = nn.LayerNorm(hidden_size)
        self.aggregate_head = nn.Linear(hidden_size, config.num_aggregates)
        self.orderby_direction_head = nn.Linear(hidden_size, config.num_directions)

        # 5. 回归头
        self.quality_head = nn.Linear(hidden_size, config.num_quality_dims)

        # scaled dot-product 的缩放因子
        self.scale = hidden_size ** 0.5

        # 小权重初始化: 让初始 logits 接近 0
        init_std = 0.02
        for head in [self.select_head, self.where_head, self.groupby_head,
                     self.orderby_col_head, self.aggregate_head,
                     self.orderby_direction_head, self.quality_head]:
            nn.init.normal_(head.weight, std=init_std)
            nn.init.zeros_(head.bias)

        # 将自定义头转为与 backbone 一致的 dtype (bfloat16)
        for m in [self.select_head, self.where_head, self.groupby_head,
                  self.orderby_col_head, self.aggregate_head,
                  self.orderby_direction_head, self.quality_head,
                  self.query_norm, self.key_norm, self.seq_norm]:
            m.to(torch.bfloat16)

    def _set_gradient_checkpointing(self, module, value=False):
        """委托给内部 decoder backbone 实现 gradient checkpointing。"""
        if hasattr(self.decoder, "gradient_checkpointing_enable") and value:
            self.decoder.gradient_checkpointing_enable()
        elif hasattr(self.decoder, "gradient_checkpointing_disable") and not value:
            self.decoder.gradient_checkpointing_disable()

    def _get_hidden_states(self, input_ids, attention_mask):
        """
        通过 decoder backbone 获取最后一层 hidden states。
        """
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # 取最后一层的 hidden states
        # outputs.hidden_states 是一个 tuple: (embedding_output, layer1, ..., layerN)
        hidden_states = outputs.hidden_states[-1]
        return hidden_states

    def _get_sequence_representation(self, hidden_states, attention_mask):
        """
        从 decoder hidden states 中提取序列表示。
        Decoder-Only 模型使用最后一个非 padding token 的表示 (替代 Encoder 的 [CLS])。

        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, seq_length)

        Returns:
            seq_repr: (batch_size, hidden_size)
        """
        batch_size = hidden_states.shape[0]
        # 找到每个序列最后一个非 padding token 的位置
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)
        seq_repr = hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lengths]
        return seq_repr

    def forward(
        self,
        input_ids,
        attention_mask,
        column_token_indices,
        labels=None,
        **kwargs
    ):
        """
        前向传播。

        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            column_token_indices: (batch_size, num_columns) - [COL] 标记在序列中的位置索引

        Returns:
            dict: 包含 select_logits, where_logits, groupby_logits, orderby_col_logits,
                  aggregate_logits, orderby_direction_logits, predicted_quality_scores
        """
        # 1. 获取 hidden states
        hidden_states = self._get_hidden_states(input_ids, attention_mask)

        # 2. 获取序列级表示 (最后一个 token)
        seq_repr = self._get_sequence_representation(hidden_states, attention_mask)

        # ======================================================
        # A. 提取列表示 (所有任务共用 column_states)
        # ======================================================
        batch_size = hidden_states.shape[0]

        # 从 hidden_states 中根据 column_token_indices 提取 [COL] 标记的表示
        col_indices_expanded = column_token_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        column_states = torch.gather(hidden_states, 1, col_indices_expanded)
        # column_states: (batch_size, num_columns, hidden_size)

        # LayerNorm 归一化 column_states, 并转置为 (batch, hidden, num_cols)
        col_normed = self.key_norm(column_states)
        col_states_t = col_normed.transpose(1, 2)

        # ======================================================
        # B. 每个列选择任务独立的 scaled dot-product (query 也做 LayerNorm)
        # ======================================================
        select_q = self.query_norm(self.select_head(seq_repr))
        where_q = self.query_norm(self.where_head(seq_repr))
        groupby_q = self.query_norm(self.groupby_head(seq_repr))
        orderby_q = self.query_norm(self.orderby_col_head(seq_repr))

        select_logits = torch.bmm(select_q.unsqueeze(1), col_states_t).squeeze(1) / self.scale
        where_logits = torch.bmm(where_q.unsqueeze(1), col_states_t).squeeze(1) / self.scale
        groupby_logits = torch.bmm(groupby_q.unsqueeze(1), col_states_t).squeeze(1) / self.scale
        orderby_col_logits = torch.bmm(orderby_q.unsqueeze(1), col_states_t).squeeze(1) / self.scale

        # ======================================================
        # C. 固定分类头 (归一化后再投影)
        # ======================================================
        seq_normed = self.seq_norm(seq_repr)
        aggregate_logits = self.aggregate_head(seq_normed)
        orderby_dir_logits = self.orderby_direction_head(seq_normed)

        # 品质分数回归 (sigmoid 约束到 [0, 1] 范围, 训练时 labels 同步归一化到 [0,1])
        predicted_quality_scores = torch.sigmoid(self.quality_head(seq_normed))

        # ======================================================
        # D. 整合输出
        # ======================================================
        return {
            "select_logits": select_logits,
            "where_logits": where_logits,
            "groupby_logits": groupby_logits,
            "orderby_col_logits": orderby_col_logits,
            "aggregate_logits": aggregate_logits,
            "orderby_direction_logits": orderby_dir_logits,
            "predicted_quality_scores": predicted_quality_scores,
        }

    def save_pretrained_with_lora(self, save_directory: str):
        """
        正确保存包含 LoRA adapter 的模型。
        - LoRA adapter 保存到 save_directory/lora_adapter/
        - 预测头权重保存到 save_directory/head_weights.pt
        - SCM config 保存到 save_directory/
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # 保存 SCM config
        self.config.save_pretrained(save_directory)

        # 保存 LoRA adapter
        if self.config.use_lora and hasattr(self.decoder, 'save_pretrained'):
            lora_dir = os.path.join(save_directory, "lora_adapter")
            self.decoder.save_pretrained(lora_dir)
            print(f"[SCM] LoRA adapter 已保存到 {lora_dir}")

        # 保存预测头权重 (不属于 LoRA，需要单独保存)
        head_state = {
            "select_head": self.select_head.state_dict(),
            "where_head": self.where_head.state_dict(),
            "groupby_head": self.groupby_head.state_dict(),
            "orderby_col_head": self.orderby_col_head.state_dict(),
            "aggregate_head": self.aggregate_head.state_dict(),
            "orderby_direction_head": self.orderby_direction_head.state_dict(),
            "quality_head": self.quality_head.state_dict(),
            "query_norm": self.query_norm.state_dict(),
            "key_norm": self.key_norm.state_dict(),
            "seq_norm": self.seq_norm.state_dict(),
        }
        head_path = os.path.join(save_directory, "head_weights.pt")
        torch.save(head_state, head_path)
        print(f"[SCM] 预测头权重已保存到 {head_path}")

    @classmethod
    def from_pretrained_with_lora(cls, load_directory: str, device: str = "auto"):
        """
        从保存目录加载包含 LoRA adapter 的 SCM 模型。

        流程:
          1. 加载 SCM config
          2. 加载 base decoder model
          3. 加载 LoRA adapter 并合并
          4. 加载预测头权重
        """
        import os

        # 1. 加载 config
        config = SCM_DecoderConfig.from_pretrained(load_directory)

        # 2. 加载 base decoder (不带 LoRA)
        config_no_lora = SCM_DecoderConfig(
            base_model_name=config.base_model_name,
            num_aggregates=config.num_aggregates,
            num_directions=config.num_directions,
            num_quality_dims=config.num_quality_dims,
            use_lora=False,  # 先不加 LoRA
        )
        model = cls(config_no_lora)

        # 3. 加载 LoRA adapter
        lora_dir = os.path.join(load_directory, "lora_adapter")
        if os.path.exists(lora_dir):
            # 加载 tokenizer 以获取 resize 后的 vocab size（包含 [COL] 等新增 token）
            tokenizer_path = load_directory
            if os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,fix_mistral_regex=True)
                model.decoder.resize_token_embeddings(len(tokenizer))
                print(f"[SCM] Token embeddings 已调整为 {len(tokenizer)} 以匹配保存的 checkpoint")

            model.decoder = PeftModel.from_pretrained(model.decoder, lora_dir)
            # 推理时合并 LoRA 权重到 base model，提升推理速度
            model.decoder = model.decoder.merge_and_unload()
            print(f"[SCM] LoRA adapter 已加载并合并: {lora_dir}")

        # 4. 加载预测头权重
        head_path = os.path.join(load_directory, "head_weights.pt")
        if os.path.exists(head_path):
            head_state = torch.load(head_path, map_location="cpu", weights_only=True)
            model.select_head.load_state_dict(head_state["select_head"])
            model.where_head.load_state_dict(head_state["where_head"])
            model.groupby_head.load_state_dict(head_state["groupby_head"])
            model.orderby_col_head.load_state_dict(head_state["orderby_col_head"])
            model.aggregate_head.load_state_dict(head_state["aggregate_head"])
            model.orderby_direction_head.load_state_dict(head_state["orderby_direction_head"])
            model.quality_head.load_state_dict(head_state["quality_head"])
            if "query_norm" in head_state:
                model.query_norm.load_state_dict(head_state["query_norm"])
                model.key_norm.load_state_dict(head_state["key_norm"])
                model.seq_norm.load_state_dict(head_state["seq_norm"])
            print(f"[SCM] 预测头权重已加载: {head_path}")

        # 更新 config 为原始 (含 LoRA 标记)
        model.config = config
        model.eval()
        return model
