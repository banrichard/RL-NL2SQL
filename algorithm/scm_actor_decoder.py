"""
SCM Actor - Decoder-Only 版本
用于 PPO 训练中的 Ray Actor，加载 Decoder-Only SCM 模型进行推理。

核心变化:
  - 使用 SCM_DecoderModel 替代 SCM_Unified_Model
  - 输入文本格式使用末尾 [COL] 标记
  - 序列表示使用最后一个 token (而非 [CLS])
"""

import ray
import torch
import re
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
from .scm_processor_decoder import SCM_DecoderConfig, SCM_DecoderModel, COL_TOKEN
from openrlhf.trainer.ray.launcher import BaseModelActor
import torch.nn.functional as F
from openrlhf.utils.deepspeed import DeepspeedStrategy
import os


def parse_schema(schema_string: str) -> dict:
    """Parses CREATE TABLE string into tables and columns."""
    if not schema_string:
        return {"tables": [], "columns": []}
    tables, columns = [], []
    table_defs = re.findall(
        r"CREATE TABLE\s+[`\"']?(\w+)[`\"']?\s*\((.*?)\)\s*;?",
        schema_string, re.DOTALL | re.IGNORECASE
    )
    for name, content in table_defs:
        tables.append(name)
        for line in content.strip().split('\n'):
            col_match = re.match(r"[`\"'](\w+)[`\"']", line.strip())
            if col_match and col_match.group(1).upper() not in ["PRIMARY", "FOREIGN", "CONSTRAINT"]:
                columns.append(f"{name}.{col_match.group(1)}")
    return {"tables": tables, "columns": columns}


def build_decoder_input_text(question: str, schema_columns: List[str], cot_history: str = "") -> str:
    """
    构建 Decoder-Only SCM 的输入文本。
    列标记 [COL] 放在序列末尾。
    """
    parts = [f"Question: {question}"]
    parts.append(f"Schema Columns: {', '.join(schema_columns)}")
    if cot_history and cot_history.strip():
        parts.append(f"CoT: {cot_history.strip()}")
    col_markers = "".join([f"{COL_TOKEN}{col}" for col in schema_columns])
    parts.append(col_markers)
    return "\n".join(parts)


@ray.remote(num_gpus=1)
class SCM_DecoderModelActor(BaseModelActor):
    """
    Decoder-Only SCM 模型的 Ray Actor。
    用于 PPO 训练过程中的实时 SCM 分析。
    优化: 使用批量推理替代逐样本逐步骤串行推理。
    """

    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain: str):
        """
        初始化 Decoder-Only SCM 模型和 tokenizer。
        """
        self._setup_distributed(strategy)

        try:
            # 1. 加载 Config
            self.config = SCM_DecoderConfig.from_pretrained(pretrain)

            # 2. 加载 Tokenizer (从 checkpoint 目录，包含 [COL] 标记)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrain)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 确保 [COL] 在 vocab 中
            if COL_TOKEN not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({"additional_special_tokens": [COL_TOKEN]})
                print(f"[{self.__class__.__name__}] Added {COL_TOKEN} to tokenizer")

            self.col_token_id = self.tokenizer.convert_tokens_to_ids(COL_TOKEN)

            # 3. 加载模型 (使用 LoRA 专用加载方法)
            if os.path.exists(os.path.join(pretrain, "lora_adapter")):
                # LoRA 模式: 加载 base model + LoRA adapter (自动 merge)
                self.model = SCM_DecoderModel.from_pretrained_with_lora(pretrain)
                print(f"[{self.__class__.__name__}] Loaded SCM with merged LoRA adapter")
            else:
                # 非 LoRA 模式: 直接加载完整模型
                self.model = SCM_DecoderModel.from_pretrained(pretrain, config=self.config)

            # 4. 用 DeepSpeed 准备模型
            self.model = self.strategy.prepare(self.model, is_rlhf=False)
            self.model.eval()

            print(f"[{self.__class__.__name__}] Decoder-Only SCM model loaded from {pretrain}")

        except Exception as e:
            print(f"[{self.__class__.__name__}] ERROR during model initialization: {e}")
            raise e

    @torch.no_grad()
    def forward(
        self,
        sequences_data: List[Dict],
        max_length: int = 2048,
        inference_batch_size: int = 32,
    ) -> List[Dict]:
        """
        对一批 rollout 序列执行 SCM 分析（批量推理优化版）。

        将所有 (sample, step) 组合展平为一个大 batch，分 mini-batch 做 GPU 推理，
        然后将结果重新组装回 per-sample 的 entropies/quality_scores 列表。

        Args:
            sequences_data: 每个元素包含 question, schema, full_text
            max_length: tokenizer 最大长度
            inference_batch_size: GPU mini-batch 大小 (根据显存调整)

        Returns:
            List[Dict]: 每个元素包含 entropies 和 quality_scores
        """
        import time as _time
        _t0 = _time.time()

        # ============================================================
        # Phase 1: 收集所有 (sample, step) 对的输入
        # ============================================================
        all_input_texts: List[str] = []
        all_num_cols: List[int] = []        # 每个输入对应的有效列数
        all_schema_cols: List[List[str]] = []  # 每个输入对应的 schema_columns
        sample_step_counts: List[int] = []  # 每个 sample 有多少个 step (含 t=0)

        for data in sequences_data:
            question = data.get("question", "")
            schema_str = data.get("schema", "")
            full_text = data.get("full_text", "")

            if not question or not schema_str or not full_text:
                sample_step_counts.append(0)
                continue

            parsed_schema = parse_schema(schema_str)
            schema_columns = parsed_schema["columns"]
            steps = self._parse_steps(full_text)

            # t=0: 初始状态 (无 CoT)
            all_input_texts.append(build_decoder_input_text(question, schema_columns, ""))
            all_num_cols.append(len(schema_columns))
            all_schema_cols.append(schema_columns)

            # t=1..N: 逐步累积 CoT
            cot_history = ""
            for step_text in steps:
                cot_history += step_text + " "
                all_input_texts.append(build_decoder_input_text(question, schema_columns, cot_history))
                all_num_cols.append(len(schema_columns))
                all_schema_cols.append(schema_columns)

            sample_step_counts.append(1 + len(steps))

        total_inputs = len(all_input_texts)
        if total_inputs == 0:
            return [{"entropies": [0.0], "quality_scores": [0.0]} for _ in sequences_data]

        print(f"[{self.__class__.__name__}] Batched SCM inference: "
              f"{len(sequences_data)} samples -> {total_inputs} (sample,step) pairs, "
              f"mini-batch={inference_batch_size}")

        # ============================================================
        # Phase 2: 分 mini-batch 做批量推理
        # ============================================================
        all_entropies = [0.0] * total_inputs
        all_qualities = [0.0] * total_inputs

        for batch_start in range(0, total_inputs, inference_batch_size):
            batch_end = min(batch_start + inference_batch_size, total_inputs)
            batch_texts = all_input_texts[batch_start:batch_end]
            batch_cols = all_schema_cols[batch_start:batch_end]
            batch_num_cols = all_num_cols[batch_start:batch_end]
            actual_bs = len(batch_texts)

            # 2a. Tokenize (padding 到 batch 内最长，而非 max_length，节省计算)
            encoding = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)

            input_ids = encoding["input_ids"]  # (B, seq_len)

            # 2b. 构建 column_token_indices (B, max_num_cols_in_batch)
            max_cols_in_batch = max(batch_num_cols) if batch_num_cols else 1
            column_token_indices = torch.zeros(
                actual_bs, max(max_cols_in_batch, 1), dtype=torch.long, device=self.model.device
            )

            for i in range(actual_bs):
                num_c = batch_num_cols[i]
                if num_c == 0:
                    continue
                col_positions = (input_ids[i] == self.col_token_id).nonzero(as_tuple=True)[0].tolist()
                if len(col_positions) < num_c:
                    col_positions.extend([0] * (num_c - len(col_positions)))
                elif len(col_positions) > num_c:
                    col_positions = col_positions[:num_c]
                column_token_indices[i, :num_c] = torch.tensor(col_positions, dtype=torch.long)

            # 2c. 模型前向传播 (批量)
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=encoding["attention_mask"],
                    column_token_indices=column_token_indices,
                )
            except Exception as e:
                print(f"[{self.__class__.__name__}] Batched forward error: {e}. "
                      f"Falling back to per-sample inference for this mini-batch.")
                # fallback: 逐个推理
                for i in range(actual_bs):
                    try:
                        ent, qual = self._get_analysis_for_step_single(
                            batch_texts[i], batch_cols[i], max_length
                        )
                        all_entropies[batch_start + i] = ent
                        all_qualities[batch_start + i] = qual
                    except Exception:
                        pass
                continue

            # 2d. 逐样本计算熵和品质分数
            for i in range(actual_bs):
                num_c = batch_num_cols[i]
                single_outputs = {}
                for key, val in outputs.items():
                    if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == actual_bs:
                        single_outputs[key] = val[i:i+1]
                    else:
                        single_outputs[key] = val

                all_entropies[batch_start + i] = self._calculate_total_entropy(single_outputs, num_c)
                all_qualities[batch_start + i] = self._calculate_quality_score(single_outputs)

        # ============================================================
        # Phase 3: 将结果重组为 per-sample 的列表
        # ============================================================
        results = []
        cursor = 0
        for sample_idx, data in enumerate(sequences_data):
            step_count = sample_step_counts[sample_idx]
            if step_count == 0:
                results.append({"entropies": [0.0], "quality_scores": [0.0]})
                continue

            entropies = all_entropies[cursor:cursor + step_count]
            quality_scores = all_qualities[cursor:cursor + step_count]
            cursor += step_count

            results.append({
                "entropies": entropies,
                "quality_scores": quality_scores,
            })

        _elapsed = _time.time() - _t0
        print(f"[{self.__class__.__name__}] Batched SCM inference done: "
              f"{total_inputs} forward items in {_elapsed:.1f}s "
              f"({total_inputs / max(_elapsed, 0.001):.1f} items/s)")

        return results

    def _parse_steps(self, full_text: str) -> List[str]:
        """解析 CoT 文本为逻辑步骤。"""
        steps = []
        parts = re.split(r'(### Step \d+:)', full_text)
        current_step = ""
        if parts[0] and not parts[0].startswith("### Step"):
            current_step = parts[0]

        for i in range(1, len(parts), 2):
            delimiter = parts[i]
            content = parts[i + 1] if (i + 1) < len(parts) else ""
            if current_step:
                steps.append(current_step.strip())
            current_step = delimiter + content

        if current_step:
            steps.append(current_step.strip())

        return steps if steps else [full_text.strip()]

    def _get_analysis_for_step_single(
        self,
        input_text: str,
        schema_columns: List[str],
        max_length: int,
    ) -> Tuple[float, float]:
        """
        单样本推理 fallback（仅在批量推理出错时使用）。
        """
        encoding = self.tokenizer(
            input_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)

        input_ids = encoding["input_ids"]
        num_cols = len(schema_columns)
        col_positions = (input_ids[0] == self.col_token_id).nonzero(as_tuple=True)[0].tolist()

        if len(col_positions) < num_cols:
            col_positions.extend([0] * (num_cols - len(col_positions)))
        elif len(col_positions) > num_cols:
            col_positions = col_positions[:num_cols]

        column_token_indices = torch.tensor([col_positions], dtype=torch.long).to(self.model.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=encoding["attention_mask"],
            column_token_indices=column_token_indices
        )

        total_entropy = self._calculate_total_entropy(outputs, num_cols)
        quality_score = self._calculate_quality_score(outputs)
        return total_entropy, quality_score

    def _calculate_total_entropy(self, outputs: Dict, num_cols: int) -> float:
        """计算模型输出的总熵。"""
        total_entropy = 0.0

        if "aggregate_logits" in outputs:
            probs = F.softmax(outputs["aggregate_logits"], dim=-1)
            total_entropy += torch.sum(-probs * torch.log(probs + 1e-9), dim=-1).item()

        if "select_logits" in outputs:
            # 对所有列选择头 (select/where/groupby/orderby_col) 计算熵
            for key in ["select_logits", "where_logits", "groupby_logits", "orderby_col_logits"]:
                if key in outputs:
                    valid_logits = outputs[key][:, :num_cols]
                    probs = torch.sigmoid(valid_logits)
                    col_entropy = -(probs * torch.log(probs + 1e-9) + (1 - probs) * torch.log(1 - probs + 1e-9))
                    total_entropy += torch.sum(col_entropy, dim=-1).item()

        if "orderby_direction_logits" in outputs:
            probs = F.softmax(outputs["orderby_direction_logits"], dim=-1)
            total_entropy += torch.sum(-probs * torch.log(probs + 1e-9), dim=-1).item()

        return max(total_entropy, 0.0)

    def _calculate_quality_score(self, outputs: Dict) -> float:
        """计算平均品质分数。"""
        if "predicted_quality_scores" in outputs:
            scores = outputs["predicted_quality_scores"][0].clamp(0, 10).tolist()
            return sum(scores) / len(scores) if scores else 0.0
        return 0.0
