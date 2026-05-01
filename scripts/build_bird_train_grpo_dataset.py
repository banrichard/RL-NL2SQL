"""
从 BIRD train.json + train_tables.json 构建 GRPO 训练数据集。
输出格式与 build_bird_ppo_dataset.py 完全一致。
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset.utils import serialize_schema

TRAIN_JSON    = "/data/banlujie/data/bird_train/train/train.json"
TABLES_JSON   = "/data/banlujie/data/bird_train/train/train_tables.json"
OUTPUT_JSONL  = "/data/banlujie/data/bird_train/train/bird_train_grpo.jsonl"

_SYSTEM_TMPL = """\
You are a helpful SQL expert assistant.
The assistant first thinks about how to write the SQL query by analyzing the question, database schema and external knowledge, then provides the final SQL query.
The reasoning process and SQL query are enclosed within <think> </think> and <answer> </answer> tags respectively, with no extra text outside these tags.
The answer must contain the SQL query within ```sql ``` tags.
Output format is mandatory:
<think>
...
</think>
<answer>
```sql
SELECT ...
```
</answer>
Keep <think> concise (<=120 words) so generation can always complete <answer>.
Never repeat the same sentence pattern in <think>. If you start repeating, stop thinking and immediately write the best SQL in <answer>.
Do not self-correct with repeated "Wait, no" loops.
Do not use <sql>...</sql>. Do not output plain-language explanation after </answer>.
When a table/column identifier contains spaces, parentheses, %, or hyphens, always wrap it in backticks, including alias-qualified references (e.g., f.`District Code`).

Database Schema:
{db_details}"""

_EVIDENCE_SEP = "\n\nExternal Knowledge:\n"


def load_tables(tables_path: str) -> dict:
    with open(tables_path, encoding="utf-8") as f:
        data = json.load(f)
    return {entry["db_id"]: entry for entry in data}


def build_chat_prompt(schema_str: str, question: str, evidence: str, db_id: str) -> str:
    system_content = _SYSTEM_TMPL.format(db_details=schema_str)
    if db_id:
        system_content = f"Database ID: {db_id}\n" + system_content
    if evidence and evidence.strip():
        system_content += _EVIDENCE_SEP + evidence.strip()

    return (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>"
    )


def main():
    print(f"Loading train data from {TRAIN_JSON} ...")
    with open(TRAIN_JSON, encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Loading schema from {TABLES_JSON} ...")
    tables_map = load_tables(TABLES_JSON)

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    skipped = 0
    written = 0
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for row in train_data:
            db_id    = row.get("db_id", "")
            question = row.get("question", "").strip()
            sql      = row.get("SQL", "").strip()
            evidence = row.get("evidence", "").strip()

            if not db_id or not question or not sql:
                skipped += 1
                continue

            schema_info = tables_map.get(db_id)
            if schema_info is None:
                skipped += 1
                continue

            try:
                schema_str = serialize_schema(schema_info)
            except Exception as e:
                print(f"  [WARN] serialize_schema failed for {db_id}: {e}")
                skipped += 1
                continue

            record = {
                "prompt":   build_chat_prompt(schema_str, question, evidence, db_id),
                "sql":      sql,
                "db_id":    db_id,
                "evidence": evidence,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Written: {written}, Skipped: {skipped}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
