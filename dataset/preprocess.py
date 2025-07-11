import json
import pysqlite3 as sqlite3
import os
import tqdm

from dataset.utils import serialize_schema, execute_sql


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
        result = execute_sql(db_path, sql_query)
        if type(result) == list:
            # 如果返回结果是列表，说明查询成功
            valid_samples.append(item)
        else:
            print(f"无效样本 #{i} (db_id: {db_id}): {result}")
            print(f"  -> 问题SQL: {sql_query}\n")
            invalid_samples.append(item)


    print("-" * 20)
    print(f"验证完成！")
    print(f"总样本数: {len(train_data)}")
    print(f"有效样本数: {len(valid_samples)}")
    print(f"无效样本数: {invalid_count}")

    # 可以选择将清洗后的有效数据保存到新文件
    clean_train_path = os.path.join(os.path.dirname(train_path),"train_clean.json")
    invalid_record_path = os.path.join(os.path.dirname(train_path), "invalid_records.json")
    with open(clean_train_path, 'w', encoding='utf-8') as f_out:
        json.dump(valid_samples, f_out, indent=4)
    with open(invalid_record_path, 'w', encoding='utf-8') as f_invalid:
        json.dump(invalid_samples, f_invalid, indent=4)

    print(
        f"已将 {len(valid_samples)} 条有效样本保存到: {clean_train_path}, {len(invalid_samples)}条无效样本已记录到: {invalid_record_path}")


if __name__ == "__main__":
    # 假设你的文件名为 train.json 和 train_table.json
    build_dataset(train_path="/home/banlujie/data/bird_train/train/train_clean.json", tables_path="/home/banlujie/data/bird_train/train/train_tables.json")
    # validate_dataset(train_path="/home/banlujie/data/bird_train/train/train.json",
    #                  db_root_path="/home/banlujie/data/bird_train/train/train_databases")
