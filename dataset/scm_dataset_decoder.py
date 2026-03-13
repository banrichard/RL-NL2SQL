"""
SCM Dataset - Decoder-Only 版本
为 Decoder-Only SCM 模型构建训练数据。

核心变化 (相比 Encoder-Only 版本):
  1. 输入格式: question + schema_ddl + cot_history + [COL]col1[COL]col2...
     (列标记放在末尾，让 decoder 的因果注意力能看到所有前文)
  2. 不再使用预计算的 column_token_indices，而是在 tokenize 后动态计算 [COL] 位置
  3. 使用 left-padding 使最后一个 token 对齐 (decoder 模型标准做法)
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import torch.nn.functional as F
from typing import List, Dict, Any

# --- 1. 元数据定义 ---
AGGREGATE_MAP = {label: i for i, label in enumerate(["UNKNOWN", "NONE", "COUNT", "AVG", "SUM", "MAX", "MIN"])}
DIRECTION_MAP = {label: i for i, label in enumerate(["UNKNOWN", "NONE", "ASC", "DESC"])}

# 特殊标记
COL_TOKEN = "[COL]"


def parse_schema_for_metadata(schema_string: str) -> dict:
    """从 CREATE TABLE 字符串中解析出表名和列名列表。"""
    tables, columns = [], []
    table_definitions = re.findall(
        r"CREATE TABLE [`\"']?(\w+)[`\"']? \((.*?)\);",
        schema_string, re.DOTALL | re.IGNORECASE
    )
    for table_name, table_content in table_definitions:
        tables.append(table_name)
        col_names_in_table = re.findall(r"[`\"'](\w+)[`\"']", table_content)
        for col_name in col_names_in_table:
            columns.append(f"{table_name}.{col_name}")
    return {"tables": tables, "columns": columns}


def parse_where_for_columns(where_conditions: list, all_db_columns: list) -> list:
    """从 WHERE 条件中解析出涉及的列名。"""
    mentioned_columns = []
    sorted_columns = sorted(all_db_columns, key=len, reverse=True)
    full_condition_string = " ".join([item.get("condition", "") for item in where_conditions])
    for col_name in sorted_columns:
        simple_col_name = col_name.split('.')[-1]
        if re.search(r'\b' + re.escape(simple_col_name) + r'\b', full_condition_string, re.IGNORECASE):
            mentioned_columns.append(col_name)
    return mentioned_columns


def build_decoder_input_text(question: str, schema_columns: List[str], cot_history: str = "") -> str:
    """
    构建 Decoder-Only SCM 的输入文本。

    格式:
      Question: {question}
      Schema Columns: col1, col2, ...
      CoT: {cot_history}
      [COL]col1[COL]col2[COL]col3...

    列标记 [COL] 放在序列末尾，确保因果注意力下列向量能看到所有上下文。
    """
    parts = [f"Question: {question}"]
    parts.append(f"Schema Columns: {', '.join(schema_columns)}")
    if cot_history and cot_history.strip():
        parts.append(f"CoT: {cot_history.strip()}")
    # 末尾追加 [COL] 标记
    col_markers = "".join([f"{COL_TOKEN}{col}" for col in schema_columns])
    parts.append(col_markers)
    return "\n".join(parts)


def find_col_token_positions(input_ids: torch.Tensor, col_token_id: int) -> List[int]:
    """
    在 tokenized 的 input_ids 中找到所有 [COL] 标记的位置。

    Args:
        input_ids: (seq_length,) 一维 tensor
        col_token_id: [COL] 标记的 token id

    Returns:
        列表，包含每个 [COL] 标记的位置索引
    """
    positions = (input_ids == col_token_id).nonzero(as_tuple=True)[0].tolist()
    return positions


# --- 2. Dataset 类 ---

class SCM_DecoderDataset(Dataset):
    """
    Decoder-Only SCM 的 PyTorch Dataset。

    关键变化:
      - 输入文本在末尾追加 [COL] 标记
      - column_token_indices 在 tokenize 后动态计算
      - 使用 right-padding (与训练时标准做法一致)
    """

    def __init__(self, data_file_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # 确保 [COL] 是 tokenizer 中的特殊标记
        if COL_TOKEN not in tokenizer.get_vocab():
            raise ValueError(
                f"Tokenizer 中不包含 '{COL_TOKEN}' 标记。"
                f"请在创建 Dataset 前调用 tokenizer.add_special_tokens({{'additional_special_tokens': ['{COL_TOKEN}']}})"
            )
        self.col_token_id = tokenizer.convert_tokens_to_ids(COL_TOKEN)

        print(f"正在加载数据集: {data_file_path}")
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取数据"):
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        labels_raw = item['labels']
        sql_intents = labels_raw['sql_intents']
        quality_scores = labels_raw['quality_scores']
        focused_intents = labels_raw['focused_intents']
        db_tables = item.get("schema_tables", [])
        db_columns = item.get("schema_columns", [])
        num_tables = len(db_tables)
        num_columns = len(db_columns)

        # --- 构建 Decoder-Only 输入文本 ---
        question = item['question']
        # 从原始 input_text 中提取 cot_history (最后一个 [SEP] 之后的内容)
        original_input = item.get('input_text', '')
        cot_parts = original_input.split('[SEP]')
        cot_history = cot_parts[-1].strip() if len(cot_parts) > 1 else ""

        input_text = build_decoder_input_text(question, db_columns, cot_history)

        # --- Tokenize ---
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].flatten()        # (seq_length,)
        attention_mask = encoding["attention_mask"].flatten()  # (seq_length,)

        # --- 动态计算 [COL] 标记位置 ---
        col_positions = find_col_token_positions(input_ids, self.col_token_id)

        # 验证: [COL] 数量应等于列数量
        if len(col_positions) != num_columns:
            # 如果被截断导致丢失了部分 [COL], 用 0 补齐 (这些位置的 logits 会在损失中被 mask)
            if len(col_positions) < num_columns:
                col_positions.extend([0] * (num_columns - len(col_positions)))
            else:
                col_positions = col_positions[:num_columns]

        column_token_indices = torch.tensor(col_positions, dtype=torch.long)

        # --- 标签转换 (与 Encoder 版本完全一致) ---

        # 1. 品质分数
        scores_tensor = torch.tensor(
            [quality_scores['convergence'], quality_scores['efficiency']],
            dtype=torch.float
        )

        # 2. 聚合函数
        agg_list = sql_intents.get('aggregate', [])
        agg_func_str = "NONE"
        if agg_list and isinstance(agg_list, list) and len(agg_list) > 0:
            for agg_item in agg_list:
                func = agg_item.get('function', 'NONE')
                if func != 'RANK':
                    agg_func_str = func
                    break
        agg_idx = AGGREGATE_MAP.get(agg_func_str, AGGREGATE_MAP["UNKNOWN"])
        agg_tensor = torch.tensor(agg_idx, dtype=torch.long)

        # 3. SELECT 列 (multi-hot)
        select_tensor = torch.zeros(num_columns)
        for col in sql_intents.get('select', []):
            if col in db_columns:
                select_tensor[db_columns.index(col)] = 1.0

        # 4. GROUP BY 列 (multi-hot)
        groupby_tensor = torch.zeros(num_columns)
        for col in sql_intents.get('group_by', []):
            if col in db_columns:
                groupby_tensor[db_columns.index(col)] = 1.0

        # 5. WHERE 列 (multi-hot)
        where_tensor = torch.zeros(num_columns)
        where_conditions = sql_intents.get('where', [])
        if where_conditions:
            mentioned_cols = parse_where_for_columns(where_conditions, db_columns)
            for col_name in mentioned_cols:
                if col_name in db_columns:
                    where_tensor[db_columns.index(col_name)] = 1.0

        # 6. FROM / JOIN (multi-hot over tables)
        from_tensor = torch.zeros(num_tables)
        for tbl in sql_intents.get('from', []):
            if tbl in db_tables:
                from_tensor[db_tables.index(tbl)] = 1.0

        join_tensor = torch.zeros(num_tables)
        for join_info in sql_intents.get('join', []):
            tbl = join_info.get('table')
            if tbl in db_tables:
                join_tensor[db_tables.index(tbl)] = 1.0

        # 7. ORDER BY
        orderby_info = sql_intents.get('order_by')
        orderby_dict = None
        if isinstance(orderby_info, list) and len(orderby_info) > 0:
            orderby_dict = orderby_info[0]
        elif isinstance(orderby_info, dict):
            orderby_dict = orderby_info

        orderby_col_idx = -100
        orderby_dir_idx = DIRECTION_MAP["NONE"]

        if orderby_dict and orderby_dict.get('column') in db_columns:
            orderby_col_idx = db_columns.index(orderby_dict['column'])
            orderby_dir_str = orderby_dict.get('direction', 'ASC').upper()
            orderby_dir_idx = DIRECTION_MAP.get(orderby_dir_str, DIRECTION_MAP["UNKNOWN"])

        orderby_col_tensor = torch.tensor(orderby_col_idx, dtype=torch.long)
        orderby_dir_tensor = torch.tensor(orderby_dir_idx, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "column_token_indices": column_token_indices,
            "labels": {
                "quality": scores_tensor,
                "aggregate": agg_tensor,
                "select": select_tensor,
                "groupby": groupby_tensor,
                "where": where_tensor,
                "from": from_tensor,
                "join": join_tensor,
                "orderby_col": orderby_col_tensor,
                "orderby_dir": orderby_dir_tensor,
                "focused_intents": focused_intents,
            }
        }


# --- 3. Collate Function ---

def scm_decoder_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Decoder-Only SCM 的 collate function。
    处理不同样本列数量不同导致的 padding 问题。
    """
    if not batch:
        return {}

    max_cols = max(len(item['column_token_indices']) for item in batch)
    max_tables = max(len(item['labels'].get('from', [])) for item in batch)

    batch_out = {"labels": {}}

    # 模型输入
    for key in ['input_ids', 'attention_mask']:
        batch_out[key] = torch.stack([item[key] for item in batch])

    # column_token_indices padding + column_mask
    padded_indices = []
    column_masks = []
    for item in batch:
        real_len = len(item['column_token_indices'])
        pad_len = max_cols - real_len
        padded_indices.append(F.pad(item['column_token_indices'], (0, pad_len)))
        # mask: 1 = 真实列, 0 = padding 列
        mask = torch.cat([torch.ones(real_len), torch.zeros(pad_len)])
        column_masks.append(mask)
    batch_out['column_token_indices'] = torch.stack(padded_indices)
    batch_out['column_mask'] = torch.stack(column_masks)  # (batch_size, max_cols)

    # 标签 padding
    labels_keys = batch[0]['labels'].keys()
    for key in labels_keys:
        if key == 'focused_intents':
            batch_out['labels'][key] = [item['labels'][key] for item in batch]
        else:
            tensors = [item['labels'][key] for item in batch]
            if key in ['select', 'groupby', 'where']:
                padded = [F.pad(t, (0, max_cols - len(t))) for t in tensors]
                batch_out['labels'][key] = torch.stack(padded)
            elif key in ['from', 'join']:
                padded = [F.pad(t, (0, max_tables - len(t))) for t in tensors]
                batch_out['labels'][key] = torch.stack(padded)
            else:
                batch_out['labels'][key] = torch.stack(tensors)

    return batch_out


# --- 4. 测试入口 ---

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B"
    data_file = "/data/banlujie/data/synsql_5k/train/enriched_scm_cot_all_full_train.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 添加 [COL] 特殊标记
    tokenizer.add_special_tokens({"additional_special_tokens": [COL_TOKEN]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SCM_DecoderDataset(data_file, tokenizer, max_length=2048)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=scm_decoder_collate_fn, shuffle=True)

    print(f"\n✅ DataLoader 构建成功！共 {len(dataset)} 条样本")

    first_batch = next(iter(dataloader))
    print("\n--- 第一个 batch 维度 ---")
    print("Inputs:")
    for key in ['input_ids', 'attention_mask', 'column_token_indices']:
        print(f"  '{key}': shape={list(first_batch[key].shape)}")
    print("Labels:")
    for key, value in first_batch['labels'].items():
        if isinstance(value, torch.Tensor):
            print(f"  '{key}': shape={list(value.shape)}")
        else:
            print(f"  '{key}': type={type(value).__name__}, len={len(value)}")
