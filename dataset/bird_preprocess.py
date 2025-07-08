import json
import sqlite3
import os
import tqdm
from collections import defaultdict


def serialize_schema(schema_info: dict) -> str:
    """
    (Final, Robust Version)
    Serializes a database schema into CREATE TABLE strings.
    This version is robust and handles inconsistent formats for the 'primary_keys' field
    (i.e., it can handle mixed lists like [1, [2, 3]]).
    """
    table_names = schema_info['table_names_original']
    column_names = schema_info['column_names_original']
    column_types = schema_info['column_types']
    primary_keys_definitions = schema_info['primary_keys']
    foreign_keys_pairs = schema_info['foreign_keys']

    # Group columns by table index
    tables = defaultdict(list)
    for i, (table_index, col_name) in enumerate(column_names):
        if table_index == -1:
            continue
        tables[table_index].append((col_name, column_types[i]))

    # --- START OF THE ROBUST FIX ---
    primary_keys_by_table_idx = defaultdict(list)

    # This loop handles both integers and lists within the primary_keys list
    for pk_definition in primary_keys_definitions:
        pk_group = []
        # Check the type to handle inconsistent data formats
        if isinstance(pk_definition, int):
            # Handles format like: 1 in [1, [2, 3]]
            pk_group = [pk_definition]
        elif isinstance(pk_definition, list):
            # Handles format like: [2, 3] in [1, [2, 3]]
            pk_group = pk_definition
        else:
            continue  # Skip if the format is unexpected

        if not pk_group:
            continue

        # By this point, pk_group is guaranteed to be a list of column indices
        first_column_index = pk_group[0]
        table_idx = column_names[first_column_index][0]

        if table_idx != -1:
            key_column_names = [column_names[col_idx][1] for col_idx in pk_group]
            primary_keys_by_table_idx[table_idx] = key_column_names
    # --- END OF THE ROBUST FIX ---

    # Create a map for foreign keys
    col_map = {i: (table_names[table_idx], col_name) for i, (table_idx, col_name) in enumerate(column_names) if
               table_idx != -1}
    foreign_keys = [(col_map[fk_col], col_map[pk_col]) for fk_col, pk_col in foreign_keys_pairs]

    # Generate CREATE TABLE statements
    create_statements = []
    for i, table_name in enumerate(table_names):
        columns_str_list = []
        for col_name, col_type in tables[i]:
            columns_str_list.append(f"  `{col_name}` {col_type.upper()}")

        pk_for_this_table = primary_keys_by_table_idx.get(i)
        if pk_for_this_table:
            pk_cols_str = ", ".join([f"`{name}`" for name in pk_for_this_table])
            columns_str_list.append(f"  PRIMARY KEY ({pk_cols_str})")

        columns_block = ",\n".join(columns_str_list)
        statement = f"CREATE TABLE `{table_name}` (\n{columns_block}\n);"
        create_statements.append(statement)

    # Add foreign key info as comments
    if foreign_keys:
        fk_comments = ["-- Foreign Keys:"]
        for (fk_table, fk_col), (pk_table, pk_col) in foreign_keys:
            fk_comments.append(f"-- `{fk_table}`.`{fk_col}` can be joined with `{pk_table}`.`{pk_col}`")
        create_statements.append("\n".join(fk_comments))

    return "\n".join(create_statements)


def build_dataset(train_path="train.json", tables_path="train_table.json",
                  output_path="/home/banlujie/data/bird_train/train/text2sql_sft_dataset.jsonl"):
    """
    构建 Text2SQL SFT 数据集。
    """
    # 1. 加载数据
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)

    # 2. 创建一个 db_id -> schema 的映射，方便查找
    schema_map = {db['db_id']: db for db in tables_data}

    # 3. 定义 Prompt 模板
    prompt_template = """You are a powerful Text-to-SQL model. Your role is to answer user questions by generating a SQL query based on the provided database schema.

    ### Database Schema:
    {schema}
    
    ### Question:
    {question}
    
    ### SQL Query:
    """

    # 4. 遍历并生成数据集
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in train_data:
            db_id = item['db_id']
            question = item['question']
            sql_query = item['SQL']

            # 检查 db_id 是否存在于 schema 定义中
            if db_id not in schema_map:
                print(f"Warning: db_id '{db_id}' not found in tables file. Skipping item.")
                continue

            # 序列化 schema
            schema_str = serialize_schema(schema_map[db_id])

            # 构建 prompt
            prompt = prompt_template.format(schema=schema_str, question=question)

            # 构建 OpenRLHF 格式的样本
            sft_sample = {
                "prompt": prompt,
                "chosen": sql_query  # 在 SFT 阶段，"chosen" 就是我们的标准答案
                # "rejected": "" # SFT阶段不需要
            }

            # 写入 JSONL 文件
            f_out.write(json.dumps(sft_sample) + "\n")

    print(f"Successfully created SFT dataset at: {output_path}")


def validate_dataset(train_path="train.json", db_root_path="train_databases"):
    """
    使用真实的 .sqlite 文件验证 train.json 中的 SQL 查询是否可以成功执行。
    """
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    valid_samples = []
    invalid_samples = []
    invalid_count = 0

    for i, item in tqdm.tqdm(enumerate(train_data)):
        db_id = item['db_id']
        sql_query = item['SQL']

        db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            print(f"警告: 数据库文件未找到，跳过验证: {db_path}")
            continue

        try:
            # 连接到数据库并执行查询
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                cursor.fetchall()  # 尝试获取结果以确保查询完整
            # 如果没有异常，则认为样本有效
            valid_samples.append(item)
        except sqlite3.Error as e:
            # 如果捕获到任何 sqlite 异常，说明 SQL 有问题
            print(f"无效样本 #{i} (db_id: {db_id}): {e}")
            print(f"  -> 问题SQL: {sql_query}\n")
            invalid_samples.append(item)
            invalid_count += 1

    print("-" * 20)
    print(f"验证完成！")
    print(f"总样本数: {len(train_data)}")
    print(f"有效样本数: {len(valid_samples)}")
    print(f"无效样本数: {invalid_count}")

    # 可以选择将清洗后的有效数据保存到新文件
    clean_train_path = "train_clean.json"
    invalid_record_path = os.path.join(os.path.dirname(train_path), "invalid_records.json")
    with open(clean_train_path, 'w', encoding='utf-8') as f_out:
        json.dump(valid_samples, f_out, indent=4)
    with open(invalid_record_path, 'w', encoding='utf-8') as f_invalid:
        json.dump(invalid_samples, f_invalid, indent=4)

    print(
        f"已将 {len(valid_samples)} 条有效样本保存到: {clean_train_path}, {len(invalid_samples)}条无效样本已记录到: {invalid_record_path}")


if __name__ == "__main__":
    # 假设你的文件名为 train.json 和 train_table.json
    # build_dataset(train_path="/home/banlujie/data/bird_train/train/train.json", tables_path="/home/banlujie/data/bird_train/train/train_tables.json")
    validate_dataset(train_path="/home/banlujie/data/bird_train/train/train.json",
                     db_root_path="/home/banlujie/data/bird_train/train/train_databases")
