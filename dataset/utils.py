import os
import pysqlite3 as sqlite3
from collections import defaultdict
import re
import signal

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


def execute_sql(db_path: str, sql: str):
    """在指定的 SQLite 資料庫上執行 SQL 並回傳結果。"""
    if not os.path.exists(db_path):
        return "Error: Database not found"
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            # 將結果轉換為可比較的格式 (例如，排序後的元組)
            return sorted(list(result))
    except Exception as e:
        return f"Error: {e}"


def extract_sql_from_output(raw_output: str) -> str:
    """
    專門從模型輸出中提取 ```sql ... ``` 程式碼區塊內的 SQL 查詢。
    """
    if not isinstance(raw_output, str):
        return ""

    raw_output = raw_output.strip()

    # 使用正規表示式尋找 ```sql ... ``` 區塊
    # re.DOTALL 允許 '.' 匹配換行符
    # re.IGNORECASE 忽略 'sql' 的大小寫
    sql_match = re.search(r"```sql\s*(.*?)\s*```", raw_output, re.DOTALL | re.IGNORECASE)

    if sql_match:
        # group(1) 獲取的是第一個括號內的匹配內容
        return sql_match.group(1).strip()
    else:
        # 如果沒有找到標準的 ```sql ... ``` 區塊，
        # 作為備用方案，可以嘗試尋找以 '--' 或 'SELECT' 開頭的程式碼片段
        # 這有助於處理模型未能完整生成 ```sql 標籤的情況
        lines = raw_output.splitlines()
        for i, line in enumerate(lines):
            line_stripped = line.strip().upper()
            if line_stripped.startswith("-- YOUR SQL QUERY") or line_stripped.startswith("SELECT"):
                # 回傳從這一行開始的所有內容
                return "\n".join(lines[i:]).strip()

    # 如果所有策略都失敗，回傳空字串
    return ""


def get_detailed_score(db_path: str, sql_to_evaluate: str, ground_truth_result: list) -> dict:
    # 現在呼叫帶有超時功能的版本
    eval_result = execute_sql_with_timeout(db_path, sql_to_evaluate, timeout_seconds=5)

    if isinstance(eval_result, str) and "Error" in eval_result:
        # 執行失敗（包括超時）
        return {"sql": sql_to_evaluate, "score": -1.0, "is_correct": False, "error": eval_result}
    else:
        # 執行成功，比較結果
        is_correct = (eval_result == ground_truth_result)
        score = 2.0 if is_correct else -1.0
        return {"sql": sql_to_evaluate, "score": score, "is_correct": is_correct, "error": None}

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    """超時信號的處理函式，直接拋出異常"""
    raise TimeoutException("SQL 執行超時！")

def execute_sql_with_timeout(db_path, sql, timeout_seconds=5):
    """
    執行 SQL，但帶有超時限制。
    """
    # 設定信號處理器
    signal.signal(signal.SIGALRM, timeout_handler)
    # 啟動一個倒數計時器，在 timeout_seconds 秒後發送 SIGALRM 信號
    signal.alarm(timeout_seconds)

    try:
        # 執行原本的 SQL 邏輯
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            result = sorted(list(cursor.fetchall()))

        # 如果成功執行，取消鬧鐘
        signal.alarm(0)
        return result
    except TimeoutException as e:
        # 如果捕獲到我們自訂的超時異常
        return f"Error: {e}"
    except Exception as e:
        # 如果是其他 SQL 執行錯誤
        signal.alarm(0)  # 同樣要取消鬧鐘
        return f"Error: {e}"