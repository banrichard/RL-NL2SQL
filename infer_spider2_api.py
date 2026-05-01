#!/usr/bin/env python3
"""
Spider2-Lite evaluation via OpenAI-compatible API.

Usage:
    python infer_spider2_api.py \\
        --model gemini-2.5-pro \\
        --api_key sk-xxx \\
        [--base_url https://api.vectorengine.ai/v1] \\
        [--filter_prefix local]   # local | bq,ga,local | all
        [--concurrency 16] [--max_tokens 8192]

Then evaluate:
    python eval_spider2.py --pred eval_res/spider2/spider2_api_<model>.json \\
        --mode local_only   # or bq / official
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

# ── SQL 提取（与 infer_api.py 保持一致）──────────────────────────────────────

SQL_START_PATTERN = re.compile(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)
SQL_CTE_START_PATTERN = re.compile(
    r"\bWITH\b\s+(?:RECURSIVE\s+)?[A-Za-z_][A-Za-z0-9_]*\s+AS\s*\(",
    re.IGNORECASE,
)


def _sanitize(text: str) -> str:
    if not text:
        return ""
    sql = text.strip()
    for marker in ("<answer>", "</answer>", "<think>", "</think>", "```sql", "```"):
        sql = sql.replace(marker, " ")
    starts = [m.start() for m in [SQL_START_PATTERN.search(sql), SQL_CTE_START_PATTERN.search(sql)] if m]
    if not starts:
        return ""
    sql = sql[min(starts):].strip()
    stop = len(sql)
    for pat in (r"</answer>", r"```", r"<\|im_end\|>", r"<\|endoftext\|>"):
        m = re.search(pat, sql, re.IGNORECASE)
        if m and m.start() < stop:
            stop = m.start()
    sql = sql[:stop].strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip() + ";"
    else:
        sql = re.split(r"\n\s*\n", sql, maxsplit=1)[0].strip()
    return sql


def extract_sql(text: str) -> str:
    for block in re.findall(r"```sql\s*(.*?)\s*```", text, re.DOTALL):
        s = _sanitize(block)
        if s:
            return s
    m = re.search(r"<sql>\s*(.*?)\s*</sql>", text, re.DOTALL | re.IGNORECASE)
    if m:
        s = _sanitize(m.group(1))
        if s:
            return s
    m = re.search(r"<answer>\s*```sql\s*(.*?)\s*```\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        s = _sanitize(m.group(1))
        if s:
            return s
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        s = _sanitize(m.group(1))
        if s:
            return s
    m = re.search(r"(\b(?:SELECT|INSERT|UPDATE|DELETE)\b[^\n;]*(?:;|$))", text, re.IGNORECASE | re.MULTILINE)
    if m:
        s = _sanitize(m.group(1))
        if s:
            return s
    return ""


# ── Schema 序列化 ─────────────────────────────────────────────────────────────

def schema_from_sqlite(db_path: str) -> str:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name")
        rows = cur.fetchall()
        conn.close()
        return "\n\n".join(sql for _, sql in rows if sql)
    except Exception as e:
        return f"-- Error reading schema: {e}"


def find_sqlite(db_name: str, db_base: Path) -> Optional[str]:
    for sub in ("spider2-localdb", "sqlite"):
        d = db_base / sub
        if d.is_dir():
            for f in d.iterdir():
                if f.name.lower() == f"{db_name.lower()}.sqlite":
                    return str(f)
    return None


def load_external_knowledge(doc_name: Optional[str], docs_dir: str) -> str:
    if not doc_name:
        return ""
    p = os.path.join(docs_dir, doc_name)
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8") as f:
        return f.read().strip()


# ── Prompt 构建 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert SQL developer specializing in data analysis across multiple database engines (BigQuery, Snowflake, SQLite).
Your task is to generate accurate, efficient SQL queries based on natural language questions and database schemas.

Key guidelines:
1. Carefully analyze the database schema before writing queries
2. Use proper JOIN conditions and avoid Cartesian products
3. Handle NULL values appropriately
4. Use the correct SQL dialect for the specified database engine
5. Ensure the query answers exactly what is asked
6. Output format:
<think>
...
</think>
<answer>
```sql
SELECT ...
```
</answer>
7. Do not output any text after </answer>
8. Keep <think> concise (<=200 words)"""

TEMPLATE = """\
Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
{db_engine}

Database ID:
{db_id}

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

{external_knowledge_section}Instructions:
- Make sure you only output the information that is asked in the question.
- The generated query should return all of the information asked in the question without any missing or extra information.
- For BigQuery: use standard SQL with fully-qualified table names (project.dataset.table).
- For SQLite: use SQLite-compatible syntax only.
- The answer must contain the SQL query within ```sql ``` tags.

Take a deep breath and think step by step to find the correct SQL query."""


def db_engine(instance_id: str) -> str:
    if instance_id.startswith(("bq", "ga")):
        return "BigQuery"
    if instance_id.startswith("local"):
        return "SQLite"
    return "Unknown"


MAX_EXT_CHARS = 8000  # external_knowledge 字符上限，防止 prompt 过长


def build_messages(item: dict, db_base: Path, docs_dir: str) -> List[dict]:
    iid = item["instance_id"]
    engine = db_engine(iid)
    db_name = item.get("db", "")

    if instance_id_is_local(iid):
        db_path = find_sqlite(db_name, db_base)
        schema_str = schema_from_sqlite(db_path) if db_path else f"-- Database: {db_name} (not found)"
    else:
        schema_str = f"-- Database: {db_name} (cloud database)"

    ext = load_external_knowledge(item.get("external_knowledge"), docs_dir)
    if len(ext) > MAX_EXT_CHARS:
        ext = ext[:MAX_EXT_CHARS] + "\n...(truncated)"
    ext_section = f"External Knowledge:\n{ext}\n\n" if ext else ""

    user_content = TEMPLATE.format(
        db_id=db_name,
        db_engine=engine,
        db_details=schema_str,
        question=item.get("question", ""),
        external_knowledge_section=ext_section,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def instance_id_is_local(iid: str) -> bool:
    return iid.startswith("local")


# ── 异步 API 调用 ─────────────────────────────────────────────────────────────

async def call_api(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[dict],
    max_tokens: int,
    temperature: float,
    idx: int,
    retries: int = 4,
) -> tuple:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}

    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.post(url, headers=headers, json=payload, timeout=180.0)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return idx, content
            except Exception as e:
                wait = 2 ** (attempt + 1)
                print(f"  [{idx}] attempt {attempt + 1} failed: {type(e).__name__}: {e}; retry in {wait}s", flush=True)
                await asyncio.sleep(wait)
        print(f"  [{idx}] all retries exhausted", flush=True)
        return idx, ""


async def run_inference(
    tasks: List[dict],
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
) -> List[str]:
    sem = asyncio.Semaphore(concurrency)
    responses = [""] * len(tasks)

    async with httpx.AsyncClient() as client:
        coros = [
            call_api(client, sem, base_url, api_key, model,
                     tasks[i]["messages"], max_tokens, temperature, i)
            for i in range(len(tasks))
        ]
        total = len(coros)
        done = 0
        for coro in asyncio.as_completed(coros):
            idx, content = await coro
            responses[idx] = content
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  Progress: {done}/{total}", flush=True)

    return responses


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_spider2(jsonl_path: str) -> List[dict]:
    data = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Spider2-Lite evaluation via OpenAI-compatible API")
    p.add_argument("--base_url", default="https://api.vectorengine.ai/v1")
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument(
        "--spider2_dir",
        default="/data/banlujie/data/spider2/spider2-lite",
    )
    p.add_argument(
        "--filter_prefix",
        default=None,
        help="只推理指定前缀的样本（逗号分隔，如 local 或 local,bq,ga），默认全部",
    )
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument(
        "--output_name",
        default=None,
        help="输出文件路径前缀（默认 eval_res/spider2/spider2_api）",
    )
    args = p.parse_args()

    if not args.api_key:
        print("ERROR: --api_key required (or set OPENAI_API_KEY)", file=sys.stderr)
        sys.exit(1)

    spider2_dir = Path(args.spider2_dir)
    jsonl_path = spider2_dir / "spider2-lite.jsonl"
    db_base = spider2_dir / "resource" / "databases"
    docs_dir = str(spider2_dir / "resource" / "documents")

    model_slug = args.model.replace("/", "_").replace(":", "_")
    output_prefix = args.output_name or "/data/banlujie/RL-NL2SQL/eval_res/spider2/spider2_api"
    output_path = f"{output_prefix}_{model_slug}.json"
    sql_out_dir = f"{output_prefix}_{model_slug}_sql"

    print("=== Spider2-Lite API Inference ===")
    print(f"Model       : {args.model}")
    print(f"Base URL    : {args.base_url}")
    print(f"Concurrency : {args.concurrency}")
    print(f"Max tokens  : {args.max_tokens}")
    print(f"Temperature : {args.temperature}")
    print(f"Output JSON : {output_path}")
    print(f"Output SQL  : {sql_out_dir}/")

    # 加载数据
    all_data = load_spider2(str(jsonl_path))
    print(f"\n加载样本: {len(all_data)} 条")

    # 前缀过滤
    if args.filter_prefix:
        prefixes = tuple(x.strip() for x in args.filter_prefix.split(",") if x.strip())
        all_data = [d for d in all_data if d["instance_id"].startswith(prefixes)]
        print(f"过滤后: {len(all_data)} 条（前缀: {prefixes}）")

    # 构建任务
    tasks = []
    for item in all_data:
        tasks.append({
            "item": item,
            "messages": build_messages(item, db_base, docs_dir),
        })
    print(f"任务数: {len(tasks)}")

    # 推理
    print("\n开始 API 推理...")
    t0 = time.time()
    responses = asyncio.run(
        run_inference(
            tasks,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
        )
    )
    elapsed = time.time() - t0
    print(f"推理完成: {elapsed:.1f}s  ({elapsed/max(len(tasks),1):.2f}s/sample)")

    # 整理结果
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(sql_out_dir, exist_ok=True)

    results = []
    for task, response in zip(tasks, responses):
        item = task["item"]
        sql = extract_sql(response)
        item["responses"] = [response]
        item["pred_sqls"] = [sql]
        item["pred_sql"] = sql
        item["db_engine"] = db_engine(item["instance_id"])
        results.append(item)

        # 写单条 .sql 文件（供官方 evaluate.py 使用）
        with open(os.path.join(sql_out_dir, f"{item['instance_id']}.sql"), "w", encoding="utf-8") as f:
            f.write(sql if sql else "-- No SQL generated")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    empty = sum(1 for r in results if not r["pred_sql"])
    print(f"\nSQL 提取: {len(results) - empty}/{len(results)} 成功  ({empty} 空)")
    print(f"JSON 结果: {output_path}")
    print(f"SQL 目录:  {sql_out_dir}/")
    print(f"\n评测命令:")
    print(f"  python eval_spider2.py --pred {output_path} --mode local_only")
    print(f"  python eval_spider2.py --pred {output_path} --mode bq")


if __name__ == "__main__":
    main()
