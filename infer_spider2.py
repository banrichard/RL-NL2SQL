#!/usr/bin/env python3
"""
Spider2-Lite 推理脚本（基于 SGLang Engine）

数据集结构：
  - instance_id 前缀 bq/ga  -> BigQuery
  - instance_id 前缀 sf     -> Snowflake
  - instance_id 前缀 local  -> SQLite (本地可执行)

本脚本为每条样本生成 SQL，输出两种格式：
  1. JSON 文件（含完整推理信息）—— eval_spider2.py 评测用
  2. 每条样本单独写一个 <instance_id>.sql 文件 —— 兼容 spider2-lite 官方 evaluate.py
"""

import argparse
import copy
import dataclasses
import json
import os
import re
import sqlite3
from collections import UserDict
from pathlib import Path
from typing import Dict, List, Optional

import sglang as sgl
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer


# ── SQL 提取 ──────────────────────────────────────────────────────────────────

SQL_START_PATTERN = re.compile(r"\b(?:SELECT|INSERT|UPDATE|DELETE|WITH)\b", re.IGNORECASE)
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

    starts = [m.start() for m in [SQL_START_PATTERN.search(sql), SQL_CTE_START_PATTERN.search(sql)] if m]
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


def extract_sql_from_output(output_text: str) -> str:
    # 优先匹配 ```sql ... ```
    for block in re.findall(r"```sql\s*(.*?)\s*```", output_text, re.DOTALL):
        s = _sanitize_sql_candidate(block)
        if s:
            return s
    # <sql>...</sql>
    m = re.search(r"<sql>\s*(.*?)\s*</sql>", output_text, re.DOTALL | re.IGNORECASE)
    if m:
        s = _sanitize_sql_candidate(m.group(1))
        if s:
            return s
    # <answer>```sql ... ```</answer>
    m = re.search(r"<answer>\s*```sql\s*(.*?)\s*```\s*</answer>", output_text, re.DOTALL | re.IGNORECASE)
    if m:
        s = _sanitize_sql_candidate(m.group(1))
        if s:
            return s
    # 任意 ``` ... ```
    m = re.search(r"```\s*(.*?)\s*```", output_text, re.DOTALL)
    if m:
        s = _sanitize_sql_candidate(m.group(1))
        if s:
            return s
    # 多行 SELECT / WITH ... AS (
    for pat in (
        r"(\bWITH\b\s+(?:RECURSIVE\s+)?[A-Za-z_][A-Za-z0-9_]*\s+AS\s*\((?:.|\n)*?)(?:\n\s*\n|\n\s*$|$)",
        r"(\b(?:SELECT|INSERT|UPDATE|DELETE)\b(?:.|\n)*?)(?:\n\s*\n|\n\s*$|$)",
    ):
        m = re.search(pat, output_text, re.IGNORECASE)
        if m:
            s = _sanitize_sql_candidate(m.group(1))
            if s and ("SELECT" in s.upper() or "INSERT" in s.upper()):
                return s
    return ""


# ── Schema 序列化 ─────────────────────────────────────────────────────────────

def serialize_schema_from_sqlite(db_path: str) -> str:
    """从 SQLite 文件读取 CREATE TABLE DDL 作为 schema。"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name")
        tables = cursor.fetchall()
        conn.close()
        return "\n\n".join(sql for _, sql in tables if sql)
    except Exception as e:
        return f"-- Error reading schema: {e}"


# ── 文档加载 ──────────────────────────────────────────────────────────────────

def load_external_knowledge(doc_name: Optional[str], documents_dir: str) -> str:
    if not doc_name:
        return ""
    doc_path = os.path.join(documents_dir, doc_name)
    if not os.path.exists(doc_path):
        return ""
    with open(doc_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_spider2_data(jsonl_path: str) -> List[dict]:
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_db_engine(instance_id: str) -> str:
    if instance_id.startswith(("bq", "ga")):
        return "BigQuery"
    if instance_id.startswith("sf"):
        return "Snowflake"
    return "SQLite"


# ── Prompt 构建 ───────────────────────────────────────────────────────────────

class SafeFormatDict(UserDict):
    def __missing__(self, key):
        return ""


def build_schema_str(item: dict, sqlite_db_dir: str) -> str:
    instance_id = item["instance_id"]
    db_name = item.get("db", "")

    if instance_id.startswith("local"):
        # 优先从 spider2-localdb 找，其次从 sqlite 找
        for subdir in ("spider2-localdb", "sqlite"):
            path = os.path.join(sqlite_db_dir, subdir, f"{db_name}.sqlite")
            if os.path.exists(path):
                return serialize_schema_from_sqlite(path)
        # 按名字模糊搜索
        for subdir in ("spider2-localdb", "sqlite"):
            d = os.path.join(sqlite_db_dir, subdir)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower() == f"{db_name.lower()}.sqlite":
                        return serialize_schema_from_sqlite(os.path.join(d, f))
        return f"-- Database: {db_name} (schema file not found)"

    # BigQuery / Snowflake / other: 没有本地 schema，返回 db 名提示
    return f"-- Database: {db_name} (cloud database, schema not available locally)"


def build_prompt(
    item: dict,
    template: str,
    tokenizer,
    system_prompt: str,
    sqlite_db_dir: str,
    documents_dir: str,
    max_prompt_tokens: int = 6000,
) -> str:
    instance_id = item["instance_id"]
    db_engine = get_db_engine(instance_id)
    schema_str = build_schema_str(item, sqlite_db_dir)

    ext_knowledge = load_external_knowledge(item.get("external_knowledge"), documents_dir)

    def _build(ext: str) -> str:
        ext_section = f"External Knowledge:\n{ext}\n\n" if ext else ""
        fmt = SafeFormatDict(
            {
                **item,
                "db_id": item.get("db", ""),
                "db_engine": db_engine,
                "db_details": schema_str,
                "question": item.get("question", ""),
                "evidence": item.get("evidence", ""),
                "external_knowledge_section": ext_section,
                "external_knowledge": ext,
            }
        )
        prompt_text = template.format_map(fmt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except ValueError:
            return f"System:\n{system_prompt}\n\nUser:\n{prompt_text}\n\nAssistant:\n"

    full_prompt = _build(ext_knowledge)

    # 如果 prompt 超长，二分截断 external_knowledge
    if ext_knowledge and len(tokenizer.encode(full_prompt)) > max_prompt_tokens:
        lo, hi = 0, len(ext_knowledge)
        while lo < hi - 64:
            mid = (lo + hi) // 2
            candidate = _build(ext_knowledge[:mid] + "\n...(truncated)")
            if len(tokenizer.encode(candidate)) <= max_prompt_tokens:
                lo = mid
            else:
                hi = mid
        full_prompt = _build(ext_knowledge[:lo] + "\n...(truncated)")

    return full_prompt


# ── SGLang 推理 ───────────────────────────────────────────────────────────────

def normalize_texts(raw_output) -> List[str]:
    if isinstance(raw_output, dict):
        text = raw_output.get("text", "")
        return [str(t) for t in text] if isinstance(text, list) else [str(text)]
    if isinstance(raw_output, list):
        texts = []
        for item in raw_output:
            if isinstance(item, dict) and "text" in item:
                v = item["text"]
                texts.extend(str(x) for x in v) if isinstance(v, list) else texts.append(str(v))
            elif isinstance(item, str):
                texts.append(item)
            else:
                texts.append(str(item))
        return texts
    return [str(raw_output)]


DEFAULT_SYSTEM_PROMPT = """You are an expert SQL developer specializing in data analysis across multiple database engines (BigQuery, Snowflake, SQLite).
Your task is to generate accurate, efficient SQL queries based on natural language questions and database schemas.

Key guidelines:
1. Carefully analyze the database schema before writing queries
2. Use proper JOIN conditions and avoid Cartesian products
3. Handle NULL values appropriately
4. Use the correct SQL dialect for the specified database engine
5. Output format is mandatory:
<think>
...
</think>
<answer>
```sql
SELECT ...
```
</answer>
6. Keep <think> concise (<=200 words) so the final <answer> is not truncated
7. Do not output any text after </answer>"""


def main():
    parser = argparse.ArgumentParser(description="Spider2-Lite SGLang 推理脚本")
    parser.add_argument(
        "--spider2_dir",
        type=str,
        default="/data/banlujie/data/spider2/spider2-lite",
        help="spider2-lite 数据集根目录",
    )
    parser.add_argument(
        "--template_path",
        type=str,
        default="/data/banlujie/RL-NL2SQL/templates/sql_generation_spider2.txt",
        help="Prompt 模板路径",
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default=None,
        help="自定义 system prompt 文件路径（可选）",
    )
    parser.add_argument(
        "--filter_prefix",
        type=str,
        default=None,
        help="只推理指定前缀的 instance（逗号分隔，如 local,bq），默认全部",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument(
        "--strategy",
        type=str,
        default="greedy",
        choices=["greedy", "voting"],
    )
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--gpu_utilization", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "--optimize_parallel_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/banlujie/RL-NL2SQL/eval_res/spider2",
        help="推理结果输出目录（JSON + 每条 .sql 文件）",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="推理 JSON 输出文件名前缀（默认由 output_dir + 模型名自动生成）",
    )
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    spider2_dir = Path(args.spider2_dir)
    jsonl_path = spider2_dir / "spider2-lite.jsonl"
    sqlite_db_dir = str(spider2_dir / "resource" / "databases")
    documents_dir = str(spider2_dir / "resource" / "documents")

    server_args = ServerArgs.from_cli_args(args)
    model_path = (
        getattr(server_args, "model_path", None)
        or getattr(server_args, "model", None)
        or getattr(server_args, "model_name", None)
    )
    if not model_path:
        raise ValueError("请通过 --model 提供模型路径")

    model_name = model_path.rstrip("/").split("/")[-1]
    os.makedirs(args.output_dir, exist_ok=True)

    # 每条 SQL 文件写入子目录（供 evaluate.py 直接读取）
    sql_out_dir = os.path.join(args.output_dir, f"sql_{model_name}")
    os.makedirs(sql_out_dir, exist_ok=True)

    output_json = args.output_name or os.path.join(args.output_dir, f"spider2_infer_{model_name}.json")

    print("=== Spider2-Lite SGLang 推理 ===")
    print(f"模型: {model_path}")
    print(f"数据集: {jsonl_path}")
    print(f"输出 JSON: {output_json}")
    print(f"输出 SQL 目录: {sql_out_dir}")

    # 加载数据
    all_data = load_spider2_data(str(jsonl_path))
    print(f"共加载 {len(all_data)} 条样本")

    # 可选过滤
    if args.filter_prefix:
        prefixes = tuple(p.strip() for p in args.filter_prefix.split(",") if p.strip())
        all_data = [d for d in all_data if d["instance_id"].startswith(prefixes)]
        print(f"过滤后剩余 {len(all_data)} 条样本（前缀: {prefixes}）")

    # 加载 tokenizer
    print("加载 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    # 加载模板
    with open(args.template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # system prompt
    if args.system_prompt_path and os.path.exists(args.system_prompt_path):
        with open(args.system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # 构建任务
    print("构建 prompts...")
    tasks = []
    for item in all_data:
        prompt = build_prompt(item, template, tokenizer, system_prompt, sqlite_db_dir, documents_dir)
        tasks.append({"prompt": prompt, "item": item})
    print(f"共 {len(tasks)} 个推理任务")

    # 采样参数
    if args.strategy == "greedy":
        sampling_params = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": args.max_tokens, "n": 1}
    else:
        t = args.temperature if args.temperature > 0 else 0.8
        sampling_params = {"temperature": t, "top_p": args.top_p, "max_new_tokens": args.max_tokens, "n": args.num_candidates}

    use_dup_sampling = (
        args.optimize_parallel_sampling
        and args.strategy == "voting"
        and args.num_candidates > 1
    )

    # 加载 SGLang Engine
    print("加载 SGLang Engine...")
    server_kwargs = dataclasses.asdict(server_args)
    server_kwargs.pop("enable_multimodal", None)
    server_kwargs["enable_multimodal"] = False
    if "mem_fraction_static" in server_kwargs:
        server_kwargs["mem_fraction_static"] = args.gpu_utilization
    llm = sgl.Engine(**server_kwargs)
    print("SGLang Engine 就绪")

    # 分批推理
    print("开始推理...")
    all_outputs = []
    for start in range(0, len(tasks), args.batch_size):
        end = min(start + args.batch_size, len(tasks))
        batch_prompts = [tasks[i]["prompt"] for i in range(start, end)]
        print(f"  处理 {start + 1} - {end} / {len(tasks)}")

        if use_dup_sampling:
            dup_prompts = []
            for p in batch_prompts:
                dup_prompts.extend([p] * args.num_candidates)
            sp = copy.deepcopy(sampling_params)
            sp["n"] = 1
            dup_out = llm.generate(dup_prompts, sp)
            batch_out = []
            for i in range(0, len(dup_out), args.num_candidates):
                cands = dup_out[i : i + args.num_candidates]
                texts = [normalize_texts(c)[0] if normalize_texts(c) else "" for c in cands]
                batch_out.append({"text": texts})
        else:
            batch_out = llm.generate(batch_prompts, sampling_params)

        all_outputs.extend(batch_out)

        # 打印末尾样本
        if batch_out:
            sample = tasks[end - 1]
            texts = normalize_texts(batch_out[-1])
            print(f"    [{sample['item']['instance_id']}] Q: {sample['item']['question'][:60]}...")
            print(f"    生成: {texts[0][:120] if texts else '(empty)'}")

    # 整理结果
    results = []
    for task, raw_out in zip(tasks, all_outputs):
        item = task["item"]
        responses = normalize_texts(raw_out)
        sqls = [extract_sql_from_output(r) for r in responses]
        pred_sql = sqls[0] if sqls else ""

        result = {
            **item,
            "db_engine": get_db_engine(item["instance_id"]),
            "responses": responses,
            "pred_sqls": sqls,
            "pred_sql": pred_sql,
        }
        results.append(result)

        # 写单独 .sql 文件（官方 evaluate.py 读取格式）
        sql_file = os.path.join(sql_out_dir, f"{item['instance_id']}.sql")
        with open(sql_file, "w", encoding="utf-8") as f:
            f.write(pred_sql if pred_sql else "-- No SQL generated")

    # 保存 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(results)
    extracted = sum(1 for r in results if r.get("pred_sql", "").strip())
    print(f"\n推理完成: {total} 条，SQL 提取成功 {extracted} 条（{extracted/total:.1%}）")
    print(f"JSON 结果: {output_json}")
    print(f"SQL 文件目录: {sql_out_dir}")


if __name__ == "__main__":
    main()
