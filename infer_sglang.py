#!/usr/bin/env python3
"""
基于 SGLang Engine 的 SQL 生成评估脚本
参考 SGLang 官方 offline_batch_inference 示例，并与 infer.py 保持尽可能一致
"""

import argparse
import dataclasses
import copy
import json
import os
import re
from collections import UserDict
from typing import List

import sglang as sgl
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

from dataset.utils import serialize_schema


SQL_START_PATTERN = re.compile(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)
SQL_CTE_START_PATTERN = re.compile(
    r"\bWITH\b\s+(?:RECURSIVE\s+)?[A-Za-z_][A-Za-z0-9_]*\s+AS\s*\(",
    re.IGNORECASE,
)


class SafeFormatDict(UserDict):
    def __missing__(self, key):
        return ""


def _sanitize_sql_candidate(text: str) -> str:
    if not text:
        return ""

    sql_candidate = text.strip()
    for marker in ("<answer>", "</answer>", "<think>", "</think>", "```sql", "```"):
        sql_candidate = sql_candidate.replace(marker, " ")

    start_matches = [m.start() for m in (SQL_START_PATTERN.search(sql_candidate), SQL_CTE_START_PATTERN.search(sql_candidate)) if m]
    if not start_matches:
        return ""
    sql_candidate = sql_candidate[min(start_matches):].strip()

    stop_idx = len(sql_candidate)
    for stop_pattern in (r"</answer>", r"```", r"<\|im_end\|>", r"<\|endoftext\|>"):
        match = re.search(stop_pattern, sql_candidate, re.IGNORECASE)
        if match and match.start() < stop_idx:
            stop_idx = match.start()
    sql_candidate = sql_candidate[:stop_idx].strip()

    if ";" in sql_candidate:
        sql_candidate = sql_candidate.split(";", 1)[0].strip() + ";"
    else:
        sql_candidate = re.split(r"\n\s*\n", sql_candidate, maxsplit=1)[0].strip()

    return sql_candidate


def build_final_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except ValueError as e:
        if "tokenizer.chat_template is not set" not in str(e):
            raise
        # Fallback for base models without chat template support.
        return (
            "System:\n"
            f"{system_prompt}\n\n"
            "User:\n"
            f"{user_prompt}\n\n"
            "Assistant:\n"
        )


def extract_sql_from_output(output_text: str) -> str:
    """
    从模型输出中提取 SQL 查询，兼容多种格式。
    """
    pattern = r"```sql\s*(.*?)\s*```"
    sql_blocks = re.findall(pattern, output_text, re.DOTALL)
    if sql_blocks:
        sql_candidate = _sanitize_sql_candidate(sql_blocks[-1])
        if sql_candidate:
            return sql_candidate

    xml_sql_pattern = r"<sql>\s*(.*?)\s*</sql>"
    match = re.search(xml_sql_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        sql_candidate = _sanitize_sql_candidate(match.group(1))
        if sql_candidate:
            return sql_candidate

    answer_pattern = r"<answer>\s*```sql\s*(.*?)\s*```\s*</answer>"
    match = re.search(answer_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        sql_candidate = _sanitize_sql_candidate(match.group(1))
        if sql_candidate:
            return sql_candidate

    code_pattern = r"```\s*(.*?)\s*```"
    match = re.search(code_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        sql_candidate = _sanitize_sql_candidate(match.group(1))
        if sql_candidate:
            return sql_candidate

    sql_line_pattern = r"(\b(?:SELECT|INSERT|UPDATE|DELETE)\b[^\n;]*(?:;|$))"
    match = re.search(sql_line_pattern, output_text, re.IGNORECASE | re.MULTILINE)
    if match:
        sql_candidate = _sanitize_sql_candidate(match.group(1))
        if sql_candidate:
            return sql_candidate

    cte_sql_pattern = r"(\bWITH\b\s+(?:RECURSIVE\s+)?[A-Za-z_][A-Za-z0-9_]*\s+AS\s*\((?:.|\n)*?)(?:\n\s*\n|\n\s*$|$)"
    match = re.search(cte_sql_pattern, output_text, re.IGNORECASE)
    if match:
        sql_candidate = _sanitize_sql_candidate(match.group(1))
        if sql_candidate and "SELECT" in sql_candidate.upper():
            return sql_candidate

    multiline_sql_pattern = r"(\b(?:SELECT|INSERT|UPDATE|DELETE)\b(?:.|\n)*?)(?:\n\s*\n|\n\s*$|$)"
    match = re.search(multiline_sql_pattern, output_text, re.IGNORECASE)
    if match:
        sql_candidate = _sanitize_sql_candidate(match.group(1))
        if sql_candidate and (
            "FROM" in sql_candidate.upper() or "INTO" in sql_candidate.upper()
        ):
            return sql_candidate

    longest_sql = None
    max_length = 0
    for line in output_text.splitlines():
        line = line.strip()
        if re.match(r"^(?:SELECT|INSERT|UPDATE|DELETE)\b", line, re.IGNORECASE):
            if len(line) > max_length:
                longest_sql = line
                max_length = len(line)

    if longest_sql and len(longest_sql.split()) > 2:
        sql_candidate = _sanitize_sql_candidate(longest_sql)
        if sql_candidate:
            return sql_candidate

    return ""


def load_data(dev_path: str, tables_path: str):
    """
    加载开发集数据和表格信息。
    """
    try:
        with open(dev_path, "r", encoding="utf-8") as f:
            dev_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Failed to load as JSON: {e}")
        print("Trying to load as JSONL format...")
        dev_data = []
        with open(dev_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        dev_data.append(json.loads(line))
                    except json.JSONDecodeError as line_error:
                        print(f"Error parsing line {line_num}: {line_error}")
        print(f"Successfully loaded {len(dev_data)} records from JSONL format")

    with open(tables_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    schema_map = {db["db_id"]: db for db in tables_data}
    return dev_data, schema_map


def load_system_prompt(system_prompt_path: str = None) -> str:
    if system_prompt_path and os.path.exists(system_prompt_path):
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        print(f"使用自定义 system prompt: {system_prompt_path}")
        return system_prompt

    print("使用默认 system prompt")
    return """You are an expert SQL developer with deep knowledge of database design and query optimization.
Your task is to generate accurate, efficient SQL queries based on natural language questions and database schemas.

Key guidelines:
1. Always analyze the database schema carefully before writing queries
2. Use proper JOIN conditions and avoid Cartesian products
3. Handle NULL values appropriately
4. Use SQLite-compatible syntax only
5. Ensure the query answers exactly what is asked, no more, no less
6. Output format is mandatory:
<think>
...
</think>
<answer>
```sql
SELECT ...
```
</answer>
7. Do not use <sql>...</sql>, and do not output any text after </answer>
8. Keep <think> concise (<=120 words) so the final <answer> is not truncated
9. If an identifier contains spaces, parentheses, %, or hyphens, always quote it with backticks, including alias-qualified usage (e.g., f.`District Code`)

Remember: Accuracy is paramount. Double-check your column names, table relationships, and logic."""


def prepare_tasks_omni(dev_data, schema_map, template_path: str, tokenizer, system_prompt: str):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    tasks = []
    for item in dev_data:
        db_id = item.get("db_id")
        if not db_id or db_id not in schema_map:
            print(f"Warning: db_id '{db_id}' not found in schema_map, skipping...")
            continue

        schema_str = item.get("db_details", "")
        if not schema_str:
            schema_str = serialize_schema(schema_map[db_id])
        question = item.get("question", "")
        db_engine = item.get("db_engine", "SQLite")
        evidence = item.get("evidence", "")
        format_values = SafeFormatDict(
            {
                **item,
                "db_id": db_id,
                "db_details": schema_str,
                "question": question,
                "db_engine": db_engine,
                "evidence": evidence,
            }
        )
        prompt_text = template.format_map(format_values)
        final_prompt = build_final_prompt(tokenizer, system_prompt, prompt_text)

        tasks.append({"prompt": final_prompt, "original_item": item})

    return tasks


def prepare_tasks(dev_data, schema_map, template_path: str, tokenizer, system_prompt: str):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    tasks = []
    for item in dev_data:
        db_id = item.get("db_id")
        if not db_id or db_id not in schema_map:
            print(f"Warning: db_id '{db_id}' not found in schema_map, skipping...")
            continue

        schema_str = serialize_schema(schema_map[db_id])
        evidence = item.get("evidence", "")
        question = item.get("question", "")
        db_engine = item.get("db_engine", "SQLite")
        prompt_text = template.format(db_id=db_id, db_details=schema_str, question=question, db_engine=db_engine, evidence=evidence)
        final_prompt = build_final_prompt(tokenizer, system_prompt, prompt_text)

        tasks.append({"prompt": final_prompt, "original_item": item})

    return tasks


def normalize_texts(raw_output) -> List[str]:
    """将 SGLang generate 的单条输出归一化为文本列表。"""
    if isinstance(raw_output, dict):
        text = raw_output.get("text", "")
        if isinstance(text, list):
            return [str(t) for t in text]
        return [str(text)]

    if isinstance(raw_output, list):
        texts = []
        for item in raw_output:
            if isinstance(item, dict) and "text" in item:
                value = item["text"]
                if isinstance(value, list):
                    texts.extend(str(v) for v in value)
                else:
                    texts.append(str(value))
            elif isinstance(item, str):
                texts.append(item)
            else:
                texts.append(str(item))
        return texts

    if isinstance(raw_output, str):
        return [raw_output]

    return [str(raw_output)]


def build_sampling_params(args, strategy: str):
    if strategy == "greedy":
        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": args.max_tokens,
            "n": 1,
        }
        print("贪心策略: temperature=0.0, n=1")
    elif strategy == "voting":
        sampling_params = {
            "temperature": 0.8 if args.temperature == 0.0 else args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_tokens,
            "n": args.num_candidates,
        }
        print(f"投票策略: temperature={sampling_params['temperature']}, n={args.num_candidates}")
    else:
        raise ValueError(f"不支持的策略: {strategy}")

    use_duplicated_sampling = (
        args.optimize_parallel_sampling
        and strategy == "voting"
        and args.num_candidates > 1
    )
    if use_duplicated_sampling:
        print(
            "启用并行采样优化: 将每条请求复制 n 次并使用 n=1 发送，"
            "避免单个大 batch + n>1 的低效路径"
        )

    return sampling_params, use_duplicated_sampling


def run_generation(tasks, llm, args, strategy: str):
    sampling_params, use_duplicated_sampling = build_sampling_params(args, strategy)

    print(f"正在进行分批推理（{strategy}）...")
    outputs = []
    for batch_start in range(0, len(tasks), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(tasks))
        batch_prompts = [tasks[i]["prompt"] for i in range(batch_start, batch_end)]

        print(f"正在处理第 {batch_start + 1} - {batch_end} 个样本...")
        if use_duplicated_sampling and len(batch_prompts) > 1:
            dup_prompts = []
            for prompt in batch_prompts:
                dup_prompts.extend([prompt] * args.num_candidates)

            single_sample_params = copy.deepcopy(sampling_params)
            single_sample_params["n"] = 1
            dup_outputs = llm.generate(dup_prompts, single_sample_params)

            batch_outputs = []
            for i in range(0, len(dup_outputs), args.num_candidates):
                cand_outputs = dup_outputs[i:i + args.num_candidates]
                cand_texts = []
                for cand_output in cand_outputs:
                    texts = normalize_texts(cand_output)
                    cand_texts.append(texts[0] if texts else "")
                batch_outputs.append({"text": cand_texts})
        else:
            batch_outputs = llm.generate(batch_prompts, sampling_params)
        outputs.extend(batch_outputs)

        print(f"\n{'=' * 60}")
        print(f"{strategy} 第 {batch_end} 个样本的生成内容:")
        if batch_outputs:
            sample_idx = batch_start + len(batch_outputs) - 1
            sample_task = tasks[sample_idx]
            sample_output = batch_outputs[-1]
            sample_texts = normalize_texts(sample_output)
            print(f"问题: {sample_task['original_item'].get('question', '')}")
            print(f"数据库: {sample_task['original_item'].get('db_id', '')}")
            print(f"生成内容: {sample_texts[0] if sample_texts else ''}")
        print(f"{'=' * 60}\n")

    return outputs


def build_results(tasks, outputs):
    results = []
    for task, output in zip(tasks, outputs):
        responses = normalize_texts(output)
        sqls = [extract_sql_from_output(response) for response in responses]

        item = copy.deepcopy(task["original_item"])
        item["responses"] = responses
        item["pred_sqls"] = sqls
        results.append(item)

    return results


def save_results(results, output_name: str, model_name: str) -> str:
    output_path = output_name + "_" + model_name + ".json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"推理完成，结果已保存: {output_path}")
    return output_path


def run_strategy(tasks, llm, args, model_name: str, strategy: str, output_name: str) -> str:
    print("")
    print(f"=== 开始执行 {strategy} 推理 ===")
    outputs = run_generation(tasks, llm, args, strategy)
    results = build_results(tasks, outputs)
    return save_results(results, output_name, model_name)


# 数据集默认路径映射
DATASET_DEFAULTS = {
    "bird_dev": {
        "dev_path": "/data/banlujie/data/bird_dev/dev.json",
        "tables_path": "/data/banlujie/data/bird_dev/dev_tables.json",
        "db_root_path": "/data/banlujie/data/bird_dev/dev_databases",
    },
    "bird_train": {
        "dev_path": "/data/banlujie/data/bird_train/train/train.json",
        "tables_path": "/data/banlujie/data/bird_train/train/train_tables.json",
        "db_root_path": "/data/banlujie/data/bird_train/train/train_databases",
    },
    "spider_dev": {
        "dev_path": "/data/banlujie/data/spider/dev.json",
        "tables_path": "/data/banlujie/data/spider/tables.json",
        "db_root_path": "/data/banlujie/data/spider/database",
    },
    "spider_test": {
        "dev_path": "/data/banlujie/data/spider/test.json",
        "tables_path": "/data/banlujie/data/spider/test_tables.json",
        "db_root_path": "/data/banlujie/data/spider/test_database",
    },
}


def main():
    parser = argparse.ArgumentParser(description="基于 SGLang Engine 的 SQL 生成评估脚本")
    parser.add_argument(
        "--dataset",
        type=str,
        default="bird_dev",
        choices=list(DATASET_DEFAULTS.keys()),
        help="数据集名称，自动设置 dev_path / tables_path / db_root_path 的默认值",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default=None,
        help="数据集路径（留空则根据 --dataset 自动设置）",
    )
    parser.add_argument(
        "--tables_path",
        type=str,
        default=None,
        help="表格信息路径（留空则根据 --dataset 自动设置）",
    )
    parser.add_argument(
        "--db_root_path",
        type=str,
        default=None,
        help="数据库文件根目录（留空则根据 --dataset 自动设置）",
    )
    parser.add_argument(
        "--template_path",
        type=str,
        default="/data/banlujie/RL-NL2SQL/templates/sql_generation.txt",
        help="prompt 模板路径",
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default=None,
        help="自定义 system prompt 文件路径（可选）",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p")
    parser.add_argument("--max_tokens", type=int, default=4096, help="最大生成 token 数")
    parser.add_argument(
        "--strategy",
        type=str,
        default="greedy",
        choices=["greedy", "voting", "compare"],
        help="生成策略: greedy(贪心)、voting(投票) 或 compare(同进程顺序执行 greedy+voting)",
    )
    parser.add_argument("--num_candidates", type=int, default=5, help="投票策略时的候选 SQL 数量")
    parser.add_argument("--gpu_utilization", type=float, default=0.6, help="GPU 利用率（0.0-1.0），用于 SGLang 显存控制")
    parser.add_argument("--batch_size", type=int, default=500, help="分批推理大小")
    parser.add_argument(
        "--optimize_parallel_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在投票模式下将 n>1 转换为复制请求 + n=1 以提升吞吐",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="/data/banlujie/RL-NL2SQL/eval_res/sglang_inference_result",
        help="推理结果输出文件路径前缀",
    )
    parser.add_argument(
        "--compare_greedy_output_name",
        type=str,
        default=None,
        help="compare 模式下 greedy 结果输出文件路径前缀",
    )
    parser.add_argument(
        "--compare_voting_output_name",
        type=str,
        default=None,
        help="compare 模式下 voting 结果输出文件路径前缀",
    )
    parser.add_argument(
        "--reasoning_parser",
        type=str,
        default="qwen3",
        help="推理解析器类型"
    )
    # 参考 SGLang 官方示例：由 ServerArgs 注入服务端/引擎参数
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    # 根据 --dataset 自动填充未指定的路径
    ds_defaults = DATASET_DEFAULTS[args.dataset]
    if args.dev_path is None:
        args.dev_path = ds_defaults["dev_path"]
    if args.tables_path is None:
        args.tables_path = ds_defaults["tables_path"]
    if args.db_root_path is None:
        args.db_root_path = ds_defaults["db_root_path"]
    server_args = ServerArgs.from_cli_args(args)

    model_path = (
        getattr(server_args, "model_path", None)
        or getattr(server_args, "model", None)
        or getattr(server_args, "model_name", None)
    )
    if not model_path:
        raise ValueError("请通过 SGLang ServerArgs 提供模型路径，例如 --model-path <path_or_hf_id>")

    print("=== 基于 SGLang Engine 的 SQL 生成评估 ===")
    print(f"模型路径: {model_path}")
    print(f"开发集路径: {args.dev_path}")
    print(f"数据库路径: {args.db_root_path}")
    print(f"GPU利用率: {args.gpu_utilization}")
    print(f"温度: {args.temperature}")
    print(f"最大tokens: {args.max_tokens}")
    print(f"策略: {args.strategy}")

    if os.path.exists(model_path):
        print(f"使用本地模型: {model_path}")
    elif model_path.startswith("/") or model_path.startswith("."):
        print(f"错误：模型路径不存在 - {model_path}")
        return
    else:
        print(f"使用 Hugging Face 模型: {model_path}")

    if not os.path.exists(args.dev_path):
        print(f"错误：开发集路径不存在 - {args.dev_path}")
        return
    if not os.path.exists(args.tables_path):
        print(f"错误：表格路径不存在 - {args.tables_path}")
        return
    if not os.path.exists(args.db_root_path):
        print(f"错误：数据库路径不存在 - {args.db_root_path}")
        return
    if not os.path.exists(args.template_path):
        print(f"错误：模板路径不存在 - {args.template_path}")
        return

    print("正在加载数据...")
    dev_data, schema_map = load_data(args.dev_path, args.tables_path)
    print(f"加载了 {len(dev_data)} 条开发集数据")

    print("正在加载 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except ValueError:
        print("Fast tokenizer 加载失败，回退到 slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    system_prompt = load_system_prompt(args.system_prompt_path)

    print("正在准备 prompts...")
    tasks = prepare_tasks(dev_data, schema_map, args.template_path, tokenizer, system_prompt)
    print(f"准备了 {len(tasks)} 个推理任务")

    print("正在加载 SGLang Engine...")
    server_kwargs = dataclasses.asdict(server_args)
    if "mem_fraction_static" in server_kwargs and "gpu_memory_utilization" not in server_kwargs:
        server_kwargs["mem_fraction_static"] = args.gpu_utilization
    server_kwargs['enable_multimodal'] = False
    llm = sgl.Engine(**server_kwargs)
    print("SGLang Engine 加载完成！")

    model_name = model_path.rstrip("/").split("/")[-1]
    if args.strategy == "compare":
        greedy_output_name = args.compare_greedy_output_name or f"{args.output_name}_greedy"
        voting_output_name = args.compare_voting_output_name or f"{args.output_name}_voting"
        greedy_output_path = run_strategy(tasks, llm, args, model_name, "greedy", greedy_output_name)
        voting_output_path = run_strategy(tasks, llm, args, model_name, "voting", voting_output_name)
        print("")
        print("compare 模式推理完成:")
        print(f"  greedy 输出: {greedy_output_path}")
        print(f"  voting 输出: {voting_output_path}")
    else:
        run_strategy(tasks, llm, args, model_name, args.strategy, args.output_name)


# The __main__ condition is necessary here because we use "spawn" to create subprocesses.
# Spawn starts a fresh program every time, if there is no __main__, it will keep spawning.
if __name__ == "__main__":
    main()
