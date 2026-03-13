import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import torch.nn.functional as F
from typing import List, Dict, Any

# --- 1. 準備工作：定義元數據和 Schema 加載 ---

# 為所有固定分類任務定義標籤到索引的映射
AGGREGATE_MAP = {label: i for i, label in enumerate(["UNKNOWN", "NONE", "COUNT", "AVG", "SUM", "MAX", "MIN"])}
DIRECTION_MAP = {label: i for i, label in enumerate(["UNKNOWN", "NONE", "ASC", "DESC"])}  # 增加 NONE 選項


def parse_schema_for_metadata(schema_string: str) -> dict:
    """
    從 CREATE TABLE 字符串中解析出表名和 "table.column" 格式的列名列表。
    返回一個包含 'tables' 和 'columns' 列表的字典。
    """
    tables = []
    columns = []

    # 使用正則表達式找到所有 CREATE TABLE 語句
    table_definitions = re.findall(r"CREATE TABLE [`\"']?(\w+)[`\"']? \((.*?)\);", schema_string,
                                   re.DOTALL | re.IGNORECASE)

    for table_name, table_content in table_definitions:
        tables.append(table_name)
        # 找到表中的所有列定義 (一個簡化的解析器)
        # 它會提取括號內所有被引號包圍的單詞
        col_names_in_table = re.findall(r"[`\"'](\w+)[`\"']", table_content)
        for col_name in col_names_in_table:
            columns.append(f"{table_name}.{col_name}")

    return {"tables": tables, "columns": columns}


def parse_where_for_columns(where_conditions: list, all_db_columns: list) -> list:
    """
    從 WHERE 條件的 "condition" 字符串中解析出所有涉及的欄位名。

    Args:
        where_conditions: 來自標籤的列表, e.g., [{"condition": "col_a > 5 AND col_b = 'x'"}]
        all_db_columns: 該數據庫的所有欄位名列表, e.g., ["tbl.col_a", "tbl.col_b", ...]

    Returns:
        一個包含所有被提及的欄位名的列表。
    """
    mentioned_columns = []

    # 將所有欄位名按長度降序排列，優先匹配長名稱 (例如 'player_id' vs 'id')
    sorted_columns = sorted(all_db_columns, key=len, reverse=True)

    full_condition_string = " ".join([item.get("condition", "") for item in where_conditions])

    for col_name in sorted_columns:
        # col_name 可能是 'table.column'，我們只取 column 部分進行匹配
        simple_col_name = col_name.split('.')[-1]

        # 使用正則表達式確保匹配到的是完整的單詞
        if re.search(r'\b' + re.escape(simple_col_name) + r'\b', full_condition_string, re.IGNORECASE):
            mentioned_columns.append(col_name)

    return mentioned_columns


# --- 2. 自訂 PyTorch Dataset 類 ---

class SCM_AnalyzerDataset(Dataset):
    def __init__(self, data_file_path, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        print(f"正在加載數據集: {data_file_path}")
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="讀取數據"):
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        db_id = item['db_id']
        input_text = item['input_text']
        labels = item['labels']
        sql_intents = labels['sql_intents']
        quality_scores = labels['quality_scores']
        focused_intents = labels['focused_intents']
        db_tables = item.get("schema_tables")
        db_columns = item.get("schema_columns")
        num_tables = len(db_tables)
        num_columns = len(db_columns)
        column_token_indices = item.get('column_token_indices', [])
        for token_idx in column_token_indices:
            assert 0 <= token_idx < self.max_length, \
                f"數據行 {idx}: column_token_index ({token_idx}) 超出最大長度 {self.max_length}"
        assert len(column_token_indices) == num_columns, \
            f"數據行 {idx}: column_token_indices 長度 ({len(column_token_indices)}) 與 schema_columns 長度 ({num_columns}) 不匹配"
        encoding = self.tokenizer.encode_plus(
            input_text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )

        # --- 將所有意圖 Labels 轉換為 Tensors ---

        # 1. 品質分數 (迴歸)
        scores_tensor = torch.tensor([quality_scores['convergence'], quality_scores['efficiency']], dtype=torch.float)

        # 2. 聚合函數 (分類)
        agg_list = sql_intents.get('aggregate', [])
        agg_func_str = "NONE"
        if agg_list and isinstance(agg_list, list) and len(agg_list) > 0:
            # 簡化邏輯：我們只預測第一個非 RANK 的聚合函數
            for agg_item in agg_list:
                func = agg_item.get('function', 'NONE')
                if func != 'RANK':
                    agg_func_str = func
                    break
        agg_idx = AGGREGATE_MAP.get(agg_func_str, AGGREGATE_MAP["UNKNOWN"])
        agg_tensor = torch.tensor(agg_idx, dtype=torch.long)

        select_tensor = torch.zeros(num_columns)
        for col in sql_intents.get('select', []):
            if col in db_columns: select_tensor[db_columns.index(col)] = 1.0

        groupby_tensor = torch.zeros(num_columns)
        for col in sql_intents.get('group_by', []):
            if col in db_columns: groupby_tensor[db_columns.index(col)] = 1.0

        where_tensor = torch.zeros(num_columns)
        where_conditions = sql_intents.get('where', [])
        if where_conditions:
            mentioned_cols_in_where = parse_where_for_columns(where_conditions, db_columns)

            # 3. 根據解析出的欄位，創建 multi-hot 標籤張量
            for col_name in mentioned_cols_in_where:
                if col_name in db_columns:
                    where_tensor[db_columns.index(col_name)] = 1.0

        # 4. FROM / JOIN (多標籤分類 over Tables)
        from_tensor = torch.zeros(num_tables)
        for tbl in sql_intents.get('from', []):
            if tbl in db_tables: from_tensor[db_tables.index(tbl)] = 1.0

        join_tensor = torch.zeros(num_tables)
        for join_info in sql_intents.get('join', []):
            tbl = join_info.get('table')
            if tbl in db_tables: join_tensor[db_tables.index(tbl)] = 1.0

        # 5. ORDER BY (兩個分類任務)
        orderby_info = sql_intents.get('order_by')

        orderby_dict = None
        if isinstance(orderby_info, list) and len(orderby_info) > 0:
            orderby_dict = orderby_info[0]
        elif isinstance(orderby_info, dict):
            orderby_dict = orderby_info

        # 將 'NONE' 情況的標籤設為 -100，這是 PyTorch 的標準 ignore_index
        orderby_col_idx = -100
        orderby_dir_idx = DIRECTION_MAP["NONE"]

        if orderby_dict and orderby_dict.get('column') in db_columns:
            orderby_col_idx = db_columns.index(orderby_dict['column'])
            # 【斷言 2】: 確保計算出的索引是有效的
            assert 0 <= orderby_col_idx < num_columns, f"數據行 {idx}: OrderBy column index ({orderby_col_idx}) 越界 (num_columns={num_columns})"

            orderby_dir_str = orderby_dict.get('direction', 'ASC').upper()
            orderby_dir_idx = DIRECTION_MAP.get(orderby_dir_str, DIRECTION_MAP["UNKNOWN"])

        orderby_col_tensor = torch.tensor(orderby_col_idx, dtype=torch.long)
        orderby_dir_tensor = torch.tensor(orderby_dir_idx, dtype=torch.long)
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "column_token_indices": torch.tensor(column_token_indices, dtype=torch.long),
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
                "focused_intents": focused_intents
            }
        }


# --- 3. 自訂 Collate Function ---

def scm_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    【修正版】自訂的 collate_fn，用於 SCM 模型的動態批次處理。
    - 正確處理 `column_token_indices` 的 padding。
    - 正確處理 `focused_intents` 作為元數據。
    - 將欄位和表相關的標籤 padding 邏輯分開。
    """
    if not batch: return {}

    max_cols = max(len(item.get('column_token_indices', [])) for item in batch)
    max_tables = max(len(item['labels'].get('from', [])) for item in batch)

    batch_out = {"labels": {}}

    # 處理模型輸入
    for key in ['input_ids', 'attention_mask']:
        batch_out[key] = torch.stack([item[key] for item in batch])

    # 【關鍵】處理 column_token_indices 的 padding
    padded_indices = [F.pad(item['column_token_indices'], (0, max_cols - len(item['column_token_indices']))) for item in
                      batch]
    batch_out['column_token_indices'] = torch.stack(padded_indices)

    # 處理標籤
    labels_batch = batch[0]['labels']
    for key in labels_batch:
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


# --- 4. 主執行程式 ---

if __name__ == "__main__":
    data_file = "/data/banlujie/data/synsql_5k/train/enriched_scm_cot_all_full_train.jsonl"
    model_name = "distilbert/distilbert-base-uncased"
    batch_size = 4

    dataset = SCM_AnalyzerDataset(data_file, AutoTokenizer.from_pretrained(model_name))
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=scm_collate_fn, shuffle=True)

    print(f"\n✅ DataLoader 構建成功！")

    print("\n--- 檢查第一個批次的數據維度 ---")
    first_batch = next(iter(dataloader))

    print("Inputs:")
    for key in ['input_ids', 'attention_mask']:
        print(f"'{key}':\t shape={list(first_batch[key].shape)}")

    print("\nLabels:")
    for key, value in first_batch['labels'].items():
        print(f"'{key}':\t shape={list(value.shape)}")
