import json
import random

from pebble import ProcessPool
from concurrent.futures import TimeoutError
import os
import tqdm
from collections import defaultdict

from utils import serialize_schema, execute_sql


def build_dataset(train_path="train.json", tables_path="train_table.json",
                  output_path="/home/banlujie/data/bird_train/train/text2sql_sft_dataset.jsonl"):
    """
    Text2SQL SFT Dataset construction。
    """
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)

    # 2. Create a schema map for quick access
    schema_map = {db['db_id']: db for db in tables_data}

    # 3. Prompt template for OpenRLHF
    with open("/home/banlujie/RL-NL2SQL/templates/sql_generation.txt", "r", encoding='utf-8') as f:
        prompt_template = f.read()

    # 4. generate the SFT dataset
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm.tqdm(train_data, desc="Building SFT dataset"):
            db_id = item['db_id']
            question = item['question']
            evidence = item.get('evidence', "")
            sql_query = item['SQL']

            # check if db_id exists in schema_map
            if db_id not in schema_map:
                print(f"Warning: db_id '{db_id}' not found in tables file. Skipping item.")
                continue

            # serialize the schema for the database
            schema_str = serialize_schema(schema_map[db_id])

            # construct prompt
            prompt = prompt_template.format(db_engine="SQLite", db_details=schema_str, question=question + evidence)

            # construct the SFT sample for OpenRLHF
            sft_sample = {
                "prompt": prompt,
                "SQL": sql_query
            }

            f_out.write(json.dumps(sft_sample) + "\n")

    print(f"Successfully created SFT dataset at: {output_path}")


def validate_single_item(args):
    item, db_root_path = args
    db_id = item['db_id']
    sql_query = item['SQL']
    db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")

    result = execute_sql(db_path, sql_query)

    if isinstance(result, list):
        # 驗證成功，返回 'valid' 狀態和原始 item
        return 'valid', item, None
    else:
        # 驗證失敗，返回 'invalid' 狀態、原始 item 和錯誤訊息
        error_message = f"(db_id: {db_id}): {result}\n -> Wrong SQL: {sql_query}\n"
        return 'invalid', item, error_message


def validate_dataset_multiprocess(train_path="train.json", db_root_path="train_databases", max_workers=None,
                                  timeout=10):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    valid_samples = []
    invalid_samples = []

    # 準備要傳遞給工作函式的參數列表
    tasks = [(item, db_root_path) for item in train_data]

    print(f"Starting validation for {len(train_data)} samples with up to {max_workers or os.cpu_count()} workers...")

    with ProcessPool(max_workers=max_workers) as pool:
        # 使用 map 方法，並設定超時時間
        future = pool.map(validate_single_item, tasks, timeout=timeout)
        iterator = future.result()

        # 使用 tqdm 顯示進度條
        for _ in tqdm.tqdm(range(len(tasks))):
            try:
                # 逐一獲取結果
                status, item, error_message = next(iterator)
                if status == 'valid':
                    valid_samples.append(item)
                else:
                    invalid_samples.append(item)
                    # 可以在這裡立即印出錯誤，或稍後統一處理
                    print(f"Invalid Sample: {error_message}")
            except StopIteration:
                break  # 迭代完成
            except TimeoutError as error:
                print(f"A task took too long: {error.args[1]}")
            except Exception as error:
                print(f"A task failed: {error}")

    print("\n" + "-" * 20)
    print("Finished!")
    print(f"# Total Samples: {len(train_data)}")
    print(f"# Valid Samples: {len(valid_samples)}")
    print(f"# Invalid Samples: {len(invalid_samples)}")

    # 寫入結果檔案
    clean_train_path = os.path.join(os.path.dirname(train_path), "sft_train_clean.json")
    invalid_record_path = os.path.join(os.path.dirname(train_path), "invalid_records.json")

    with open(clean_train_path, 'w', encoding='utf-8') as f_out:
        json.dump(valid_samples, f_out, indent=4, ensure_ascii=False)
    with open(invalid_record_path, 'w', encoding='utf-8') as f_invalid:
        json.dump(invalid_samples, f_invalid, indent=4, ensure_ascii=False)

    print(
        f"Saved {len(valid_samples)} valid samples to: {clean_train_path}\n"
        f"Saved {len(invalid_samples)} invalid samples to: {invalid_record_path}"
    )


def split_dataset_by_db(input_train_file="train.json", eval_db_ratio=0.2, output_train_file="sft_train.json",
                        output_eval_file="sft_eval.json"):
    print(f"Reading from '{input_train_file}'.")
    with open(input_train_file, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    # 1. 按 db_id 將所有資料分組
    data_by_db = defaultdict(list)
    for item in full_data:
        data_by_db[item['db_id']].append(item)

    # 2. 獲取所有唯一的 db_id 並隨機打亂
    all_db_ids = list(data_by_db.keys())
    random.shuffle(all_db_ids)

    # 3. 分割 db_id 列表
    split_index = int(len(all_db_ids) * (1 - eval_db_ratio))
    train_db_ids = set(all_db_ids[:split_index])
    eval_db_ids = set(all_db_ids[split_index:])

    print(f"Total #: {len(all_db_ids)}")
    print(f"Train #: {len(train_db_ids)}")
    print(f"Eval #: {len(eval_db_ids)}")

    # 4. 根據分割好的 db_id 建立新的資料集
    train_split = []
    eval_split = []

    for db_id, items in data_by_db.items():
        if db_id in train_db_ids:
            train_split.extend(items)
        else:
            eval_split.extend(items)

    print(f"Total #: {len(full_data)}")
    print(f"Train #: {len(train_split)}")
    print(f"Eval #: {len(eval_split)}")

    # 寫入檔案
    with open(output_train_file, 'w', encoding='utf-8') as f:
        json.dump(train_split, f, indent=2, ensure_ascii=False)
    print(f"New Train set is stored in'{output_train_file}'")

    with open(output_eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_split, f, indent=2, ensure_ascii=False)
    print(f"New Eval set is stored in '{output_eval_file}'")


if __name__ == "__main__":
    # validate_dataset_multiprocess(train_path="/home/banlujie/data/bird_train/train/train.json",
    #                  db_root_path="/home/banlujie/data/bird_train/train/train_databases",max_workers=8, timeout=10)
    split_dataset_by_db(input_train_file="/home/banlujie/data/bird_train/train/sft_train_clean.json",
                        eval_db_ratio=0.05,
                        output_train_file="/home/banlujie/data/bird_train/train/final_train_sft.json",
                        output_eval_file="/home/banlujie/data/bird_train/train/final_eval_sft.json")
    build_dataset(train_path="/home/banlujie/data/bird_train/train/final_train_sft.json",
                  tables_path="/home/banlujie/data/bird_train/train/train_tables.json",
                  output_path="/home/banlujie/data/bird_train/train/train_sft.json")
    build_dataset(train_path="/home/banlujie/data/bird_train/train/final_eval_sft.json",
                  tables_path="/home/banlujie/data/bird_train/train/train_tables.json",
                  output_path="/home/banlujie/data/bird_train/train/eval_sft.json")
