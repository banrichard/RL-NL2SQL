#!/usr/bin/env python3
"""
Closed-source model BIRD/Spider evaluation via OpenAI-compatible API.

Usage:
    python infer_api.py [--dataset spider_dev|spider_test|bird_dev|bird_train] [--output_name PATH] \
                        [--model MODEL_NAME] [--max_tokens N] [--temperature T] \
                        [--concurrency N] [--base_url URL] [--api_key KEY] \
                        [--strategy greedy|voting] [--num_candidates N]

Then evaluate:
    python eval_bird.py --pred <output>.json --gold /data/.../dev.json \
                        --db_path /data/.../dev_databases --mode greedy_search
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

from dataset.utils import serialize_schema

# --------------------------------------------------------------------------- #
# SQL extraction (identical logic to infer.py)
# --------------------------------------------------------------------------- #

SQL_START_PATTERN = re.compile(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)
SQL_CTE_START_PATTERN = re.compile(
    r"\bWITH\b\s+(?:RECURSIVE\s+)?[A-Za-z_][A-Za-z0-9_]*\s+AS\s*\(",
    re.IGNORECASE,
)


def _sanitize_sql_candidate(text: str) -> str:
    if not text:
        return ""
    sql = text.strip()
    for marker in ("<answer>", "</answer>", "<think>", "</think>", "```sql", "```"):
        sql = sql.replace(marker, " ")

    starts = [
        m.start()
        for m in (SQL_START_PATTERN.search(sql), SQL_CTE_START_PATTERN.search(sql))
        if m
    ]
    if not starts:
        return ""
    sql = sql[min(starts):].strip()

    stop_idx = len(sql)
    for pat in (r"</answer>", r"```", r"<\|im_end\|>", r"<\|endoftext\|>"):
        m = re.search(pat, sql, re.IGNORECASE)
        if m and m.start() < stop_idx:
            stop_idx = m.start()
    sql = sql[:stop_idx].strip()

    if ";" in sql:
        sql = sql.split(";", 1)[0].strip() + ";"
    else:
        sql = re.split(r"\n\s*\n", sql, maxsplit=1)[0].strip()
    return sql


def extract_sql_from_output(text: str) -> str:
    # ```sql ... ```
    for block in re.findall(r"```sql\s*(.*?)\s*```", text, re.DOTALL):
        cand = _sanitize_sql_candidate(block)
        if cand:
            return cand

    # <sql> ... </sql>
    m = re.search(r"<sql>\s*(.*?)\s*</sql>", text, re.DOTALL | re.IGNORECASE)
    if m:
        cand = _sanitize_sql_candidate(m.group(1))
        if cand:
            return cand

    # <answer>```sql ... ```</answer>
    m = re.search(r"<answer>\s*```sql\s*(.*?)\s*```\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        cand = _sanitize_sql_candidate(m.group(1))
        if cand:
            return cand

    # ``` ... ```
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        cand = _sanitize_sql_candidate(m.group(1))
        if cand:
            return cand

    # bare SELECT / INSERT / ...
    m = re.search(r"(\b(?:SELECT|INSERT|UPDATE|DELETE)\b[^\n;]*(?:;|$))", text, re.IGNORECASE | re.MULTILINE)
    if m:
        cand = _sanitize_sql_candidate(m.group(1))
        if cand:
            return cand

    return ""


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are an expert SQL developer with deep knowledge of database design and "
    "query optimization. Your task is to generate accurate, efficient SQL queries "
    "based on natural language questions and database schemas.\n\n"
    "Key guidelines:\n"
    "1. Always analyze the database schema carefully before writing queries\n"
    "2. Use proper JOIN conditions and avoid Cartesian products\n"
    "3. Handle NULL values appropriately\n"
    "4. Use SQLite-compatible syntax only\n"
    "5. Ensure the query answers exactly what is asked, no more, no less\n"
    "6. Output format is mandatory:\n"
    "<think>\n...\n</think>\n<answer>\n```sql\nSELECT ...\n```\n</answer>\n"
    "7. Do not use <sql>...</sql>, and do not output any text after </answer>\n"
    "8. Keep <think> concise (<=120 words) so the final <answer> is not truncated\n"
    "9. If an identifier contains spaces, parentheses, %, or hyphens, always quote "
    "it with backticks\n\n"
    "Remember: Accuracy is paramount. Double-check your column names, table "
    "relationships, and logic."
)

TEMPLATE = (
    "Task Overview:\n"
    "You are a data science expert. Below, you are provided with a database schema "
    "and a natural language question. Your task is to understand the schema and "
    "generate a valid SQL query to answer the question.\n\n"
    "Database Engine:\nSQLite\n\n"
    "Database ID:\n{db_id}\n\n"
    "Database Schema:\n{db_details}\n"
    "This schema describes the database's structure, including tables, columns, "
    "primary keys, foreign keys, and any relevant relationships or constraints.\n\n"
    "Question:\n{question}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Instructions:\n"
    "- Make sure you only output the information that is asked in the question.\n"
    "- The generated query should return all of the information asked in the question "
    "without any missing or extra information.\n"
    "- The answer must contain the SQL query within ```sql ``` tags.\n\n"
    "Take a deep breath and think step by step to find the correct SQL query."
)


def load_data(dev_path: str, tables_path: str):
    with open(dev_path, encoding="utf-8") as f:
        try:
            dev_data = json.load(f)
        except json.JSONDecodeError:
            dev_data = [json.loads(l) for l in f if l.strip()]

    with open(tables_path, encoding="utf-8") as f:
        tables_data = json.load(f)

    schema_map = {db["db_id"]: db for db in tables_data}
    return dev_data, schema_map


def build_messages(item: dict, schema_map: dict) -> list[dict]:
    db_id = item["db_id"]
    schema_str = serialize_schema(schema_map[db_id])
    question = item["question"]
    evidence = item.get("evidence", "")

    user_content = TEMPLATE.format(
        db_id=db_id,
        db_details=schema_str,
        question=question,
        evidence=evidence,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --------------------------------------------------------------------------- #
# Async API caller
# --------------------------------------------------------------------------- #

async def call_api(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    idx: int,
    candidate_idx: int = 0,
    retries: int = 3,
) -> tuple[int, int, str]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request_tag = f"[{idx}:{candidate_idx}]"

    async with semaphore:
        for attempt in range(retries):
            try:
                resp = await client.post(url, headers=headers, json=payload, timeout=120.0)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return idx, candidate_idx, content
            except Exception as e:
                wait = 2 ** attempt
                print(f"{request_tag} attempt {attempt + 1} failed: {e}; retrying in {wait}s", flush=True)
                await asyncio.sleep(wait)
        print(f"{request_tag} all retries exhausted, returning empty", flush=True)
        return idx, candidate_idx, ""


async def run_inference(
    tasks: list[dict],
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    strategy: str,
    num_candidates: int,
) -> list[list[str]]:
    semaphore = asyncio.Semaphore(concurrency)
    request_count_per_task = 1 if strategy == "greedy" else num_candidates
    results = [[""] * request_count_per_task for _ in range(len(tasks))]

    async with httpx.AsyncClient() as client:
        coros = [
            call_api(
                client, semaphore, base_url, api_key, model,
                tasks[i]["messages"], max_tokens, temperature, i, candidate_idx,
            )
            for i in range(len(tasks))
            for candidate_idx in range(request_count_per_task)
        ]
        total = len(coros)
        done = 0
        for coro in asyncio.as_completed(coros):
            idx, candidate_idx, content = await coro
            results[idx][candidate_idx] = content
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  Progress: {done}/{total}", flush=True)

    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="BIRD/Spider evaluation with closed-source API model")
    p.add_argument("--base_url", default="https://api.vectorengine.ai/v1",
                   help="OpenAI-compatible API base URL (must include /v1)")
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""),
                   help="API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--model", default="gemini-2.5-pro",
                   help="Model name to request")
    p.add_argument("--dataset", default="spider_dev",
                   choices=["bird_dev", "bird_train", "spider_dev", "spider_test"],
                   help="Dataset to evaluate")
    p.add_argument("--dev_path", default=None, help="Override dev.json path")
    p.add_argument("--tables_path", default=None, help="Override tables.json path")
    p.add_argument("--output_name", default=None,
                   help="Output file prefix (default: eval_res/eval_results_api_<dataset>)")
    p.add_argument("--strategy", default="greedy", choices=["greedy", "voting"],
                   help="Inference strategy to run")
    p.add_argument("--num_candidates", type=int, default=8,
                   help="Number of candidates per sample for voting strategy")
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=16,
                   help="Number of concurrent API requests")
    return p.parse_args()


DATASET_DEFAULTS = {
    "bird_dev": {
        "dev_path":    "/data/banlujie/data/bird_dev/dev.json",
        "tables_path": "/data/banlujie/data/bird_dev/dev_tables.json",
        "db_path":     "/data/banlujie/data/bird_dev/dev_databases",
    },
    "bird_train": {
        "dev_path":    "/data/banlujie/data/bird_train/train/train.json",
        "tables_path": "/data/banlujie/data/bird_train/train/train_tables.json",
        "db_path":     "/data/banlujie/data/bird_train/train/train_databases",
    },
    "spider_dev": {
        "dev_path":    "/data/banlujie/data/spider/dev.json",
        "tables_path": "/data/banlujie/data/spider/tables.json",
        "db_path":     "/data/banlujie/data/spider/database",
    },
    "spider_test": {
        "dev_path":    "/data/banlujie/data/spider/test.json",
        "tables_path": "/data/banlujie/data/spider/test_tables.json",
        "db_path":     "/data/banlujie/data/spider/test_database",
    },
}


def main():
    args = parse_args()

    if not args.api_key:
        print("ERROR: --api_key is required (or set OPENAI_API_KEY)", file=sys.stderr)
        sys.exit(1)
    if args.num_candidates < 1:
        print("ERROR: --num_candidates must be >= 1", file=sys.stderr)
        sys.exit(1)

    ds = DATASET_DEFAULTS[args.dataset]
    dev_path    = args.dev_path    or ds["dev_path"]
    tables_path = args.tables_path or ds["tables_path"]
    db_path     = ds["db_path"]

    model_slug = args.model.replace("/", "_").replace(":", "_")
    output_prefix = (
        args.output_name
        or (
            f"/data/banlujie/RL-NL2SQL/eval_res/eval_results_api_{args.dataset}"
            if args.strategy == "greedy"
            else f"/data/banlujie/RL-NL2SQL/eval_res/eval_results_api_voting_{args.dataset}"
        )
    )
    output_path = f"{output_prefix}_{model_slug}.json"
    eval_mode = "greedy_search" if args.strategy == "greedy" else "major_voting"

    print(f"=== API Evaluation ===")
    print(f"Model      : {args.model}")
    print(f"Base URL   : {args.base_url}")
    print(f"Dataset    : {args.dataset}")
    print(f"Strategy   : {args.strategy}")
    if args.strategy == "voting":
        print(f"Candidates : {args.num_candidates}")
    print(f"dev_path   : {dev_path}")
    print(f"concurrency: {args.concurrency}")
    print(f"max_tokens : {args.max_tokens}")
    print(f"temperature: {args.temperature}")
    print(f"output     : {output_path}")

    # Load data
    print("\nLoading data...")
    dev_data, schema_map = load_data(dev_path, tables_path)
    print(f"Loaded {len(dev_data)} samples")

    # Build tasks
    tasks = []
    skipped = 0
    for item in dev_data:
        db_id = item.get("db_id", "")
        if not db_id or db_id not in schema_map:
            print(f"  WARNING: db_id '{db_id}' not in schema_map, skipping")
            skipped += 1
            continue
        tasks.append({"item": item, "messages": build_messages(item, schema_map)})

    print(f"Tasks built: {len(tasks)} (skipped {skipped})")

    # Run inference
    print("\nRunning inference...")
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
            strategy=args.strategy,
            num_candidates=args.num_candidates,
        )
    )
    elapsed = time.time() - t0
    avg_elapsed = (elapsed / len(tasks)) if tasks else 0.0
    print(f"Inference done in {elapsed:.1f}s ({avg_elapsed:.2f}s/sample)")

    # Build result list (same format as infer.py so eval_bird.py works unchanged)
    results = []
    for task, response_group in zip(tasks, responses):
        item = task["item"]
        sqls = [extract_sql_from_output(response) for response in response_group]
        item["responses"] = response_group
        item["pred_sqls"] = sqls
        results.append(item)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # Quick extraction stats
    total_sqls = sum(len(r["pred_sqls"]) for r in results)
    empty = sum(1 for r in results for sql in r["pred_sqls"] if not sql)
    print(f"SQL extracted: {total_sqls - empty}/{total_sqls} ({empty} empty)")

    print(f"\nTo evaluate, run:")
    print(
        f"  python eval_bird.py --pred {output_path} "
        f"--gold {dev_path} --db_path {db_path} --mode {eval_mode}"
    )


if __name__ == "__main__":
    main()
