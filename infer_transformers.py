#!/usr/bin/env python3
"""
基于 Transformers 原生 generate 的 SQL 生成脚本
用于替代 vLLM 推理，输出格式与 infer.py 保持兼容
"""

import argparse
import json
import os
import re
import shutil
import tempfile
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from accelerate import Accelerator
    from accelerate import FullyShardedDataParallelPlugin
    from accelerate.utils import gather_object

    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

from dataset.utils import serialize_schema

# 尝试导入peft（用于LoRA合并）
try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("警告: peft 库未安装，无法使用LoRA功能")


def merge_lora_adapter(base_model_path, lora_adapter_path, output_dir=None):
    """
    合并LoRA adapter到基础模型
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft 库未安装，请运行: pip install peft")

    print("开始合并 LoRA adapter...")
    print(f"  基础模型: {base_model_path}")
    print(f"  LoRA adapter: {lora_adapter_path}")

    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("加载 LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("合并权重...")
    merged_model = model.merge_and_unload()

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="merged_lora_")

    os.makedirs(output_dir, exist_ok=True)
    print(f"保存合并后的模型到: {output_dir}")

    merged_model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    for file_name in ["generation_config.json", "chat_template.jinja", "preprocessor_config.json"]:
        src = os.path.join(base_model_path, file_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, file_name))

    print(f"LoRA 合并完成！合并后的模型路径: {output_dir}")
    return output_dir


def extract_sql_from_output(output_text):
    """
    从模型输出中提取 SQL 查询
    支持多种格式的 SQL 代码块
    """
    pattern = r"```sql\s*(.*?)\s*```"
    sql_blocks = re.findall(pattern, output_text, re.DOTALL)
    if sql_blocks:
        return sql_blocks[-1].strip()

    xml_sql_pattern = r"<sql>\s*(.*?)\s*</sql>"
    match = re.search(xml_sql_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    answer_pattern = r"<answer>\s*```sql\s*(.*?)\s*```\s*</answer>"
    match = re.search(answer_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    code_pattern = r"```\s*(.*?)\s*```"
    match = re.search(code_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        sql_candidate = match.group(1).strip()
        if any(keyword in sql_candidate.upper() for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
            return sql_candidate

    sql_line_pattern = r"((?:SELECT|INSERT|UPDATE|DELETE)(?:[^;])+(?:;|$))"
    match = re.search(sql_line_pattern, output_text, re.IGNORECASE | re.MULTILINE)
    if match:
        sql_candidate = match.group(1).strip()
        if len(sql_candidate.split()) > 1:
            return sql_candidate

    multiline_sql_pattern = r"((?:SELECT|INSERT|UPDATE|DELETE)(?:.|\n)*?)(?:\n\s*\n|\n\s*$|$)"
    match = re.search(multiline_sql_pattern, output_text, re.IGNORECASE)
    if match:
        sql_candidate = match.group(1).strip()
        if len(sql_candidate.split()) > 2 and (
            "FROM" in sql_candidate.upper() or "INTO" in sql_candidate.upper()
        ):
            return sql_candidate

    longest_sql = None
    max_length = 0
    for line in output_text.splitlines():
        line = line.strip()
        if line.upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE")):
            if len(line) > max_length:
                longest_sql = line
                max_length = len(line)

    if longest_sql and len(longest_sql.split()) > 2:
        return longest_sql

    return ""


def load_data(dev_path, tables_path):
    """
    加载开发集数据和表格信息
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


def load_system_prompt(system_prompt_path=None):
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
6. Follow the XML output format strictly: <sql>your_query_here</sql>
7. Think step by step and show your reasoning in <think> tags

Remember: Accuracy is paramount. Double-check your column names, table relationships, and logic."""


def prepare_prompts_omni(dev_data, schema_map, template_path, tokenizer, system_prompt):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    tasks = []
    for item in dev_data:
        db_id = item.get("db_id")
        if not db_id or db_id not in schema_map:
            print(f"Warning: db_id '{db_id}' not found in schema_map, skipping...")
            continue

        schema_str = item.get("db_details", "")
        question = item.get("question", "")

        prompt_text = template.format(db_id=db_id, db_details=schema_str, question=question)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        tasks.append({"prompt": final_prompt, "original_item": item})

    return tasks


def prepare_prompts(dev_data, schema_map, template_path, tokenizer, system_prompt):
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
        question = item.get("question", "") + evidence

        prompt_text = template.format(db_id=db_id, db_details=schema_str, question=question)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        tasks.append({"prompt": final_prompt, "original_item": item})

    return tasks


def resolve_dtype(dtype_name: str):
    dtype_name = (dtype_name or "bfloat16").lower()
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def batched_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
    do_sample: bool,
    device: torch.device,
):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_lens = attention_mask.sum(dim=1).tolist()

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.inference_mode():
        sequences = model.generate(**generate_kwargs)

    # 还原为 [batch_size, num_return_sequences]
    all_texts = []
    expanded_prompt_lens = []
    for plen in prompt_lens:
        expanded_prompt_lens.extend([plen] * num_return_sequences)

    for seq, prompt_len in zip(sequences, expanded_prompt_lens):
        gen_tokens = seq[prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        all_texts.append(text.strip())

    grouped_texts = []
    cursor = 0
    for _ in range(len(prompts)):
        grouped_texts.append(all_texts[cursor: cursor + num_return_sequences])
        cursor += num_return_sequences

    return grouped_texts


def main():
    parser = argparse.ArgumentParser(description="基于 Transformers 的 SQL 生成评估脚本")
    parser.add_argument("--model_path", type=str, default="/data/banlujie/RL-NL2SQL/ckpt/phi4_mini_reasoning", help="模型路径")
    parser.add_argument("--base_model_path", type=str, default=None, help="基础模型路径（用于LoRA加载）")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA adapter路径（如果提供，将合并到模型中）")
    parser.add_argument("--merged_model_output", type=str, default=None, help="合并后的模型保存路径（如果None则使用临时目录）")

    parser.add_argument("--dev_path", type=str, default="/data/banlujie/data/bird_dev/dev.json", help="开发集路径")
    parser.add_argument("--tables_path", type=str, default="/data/banlujie/data/bird_dev/dev_tables.json", help="表格信息路径")
    parser.add_argument("--db_root_path", type=str, default="/data/banlujie/data/bird_dev/dev_databases", help="数据库文件根目录")
    parser.add_argument("--template_path", type=str, default="/data/banlujie/RL-NL2SQL/templates/sql_generation.txt", help="prompt 模板路径")
    parser.add_argument("--system_prompt_path", type=str, default=None, help="自定义 system prompt 文件路径（可选）")

    parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy", "voting"], help="生成策略: greedy(贪心) 或 voting(投票)")
    parser.add_argument("--num_candidates", type=int, default=5, help="投票策略时的候选SQL数量")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p 采样")
    parser.add_argument("--max_tokens", type=int, default=4096, help="最大生成token数")
    parser.add_argument("--batch_size", type=int, default=2, help="Transformers推理批大小")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="模型dtype")
    parser.add_argument("--use_accelerate_fsdp", action="store_true", help="使用 accelerate + FSDP 进行分布式推理")
    parser.add_argument("--fsdp_min_num_params", type=int, default=10000000, help="FSDP 自动wrap最小参数量")
    parser.add_argument("--cpu_offload", action="store_true", help="FSDP CPU offload")

    parser.add_argument("--output_name", type=str, default="/data/banlujie/RL-NL2SQL/eval_res/transformers_inference_result", help="推理结果输出文件路径")
    args = parser.parse_args()

    print("=== 基于 Transformers 原生 generate 的 SQL 推理 ===")
    print(f"模型路径: {args.model_path}")
    print(f"开发集路径: {args.dev_path}")
    print(f"数据库路径: {args.db_root_path}")
    print(f"策略: {args.strategy}")
    print(f"batch_size: {args.batch_size}")
    print(f"dtype: {args.torch_dtype}")
    print(f"use_accelerate_fsdp: {args.use_accelerate_fsdp}")

    accelerator = None
    is_main_process = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_accelerate_fsdp:
        if not ACCELERATE_AVAILABLE:
            print("错误：未安装 accelerate，请先安装: pip install accelerate")
            return

        fsdp_plugin = FullyShardedDataParallelPlugin(
            use_orig_params=True,
            cpu_offload=args.cpu_offload,
            min_num_params=max(0, args.fsdp_min_num_params),
        )
        mp = "bf16" if args.torch_dtype == "bfloat16" else ("fp16" if args.torch_dtype == "float16" else "no")
        accelerator = Accelerator(fsdp_plugin=fsdp_plugin, mixed_precision=mp)
        is_main_process = accelerator.is_main_process
        device = accelerator.device

        if is_main_process:
            print(
                f"Accelerate initialized: world_size={accelerator.num_processes}, "
                f"process_index={accelerator.process_index}, device={device}"
            )

    actual_model_path = args.model_path

    if args.lora_path:
        if not args.base_model_path:
            print("错误：使用 LoRA adapter 需要指定 --base_model_path")
            return
        if not os.path.exists(args.base_model_path):
            print(f"错误：基础模型路径不存在 - {args.base_model_path}")
            return
        if not os.path.exists(args.lora_path):
            print(f"错误：LoRA adapter 路径不存在 - {args.lora_path}")
            return

        try:
            actual_model_path = merge_lora_adapter(args.base_model_path, args.lora_path, args.merged_model_output)
        except Exception as e:
            print(f"错误：合并 LoRA adapter 失败 - {e}")
            return
    else:
        if os.path.exists(args.model_path):
            print(f"使用本地模型: {args.model_path}")
        elif args.model_path.startswith("/") or args.model_path.startswith("."):
            print(f"错误：模型路径不存在 - {args.model_path}")
            return
        else:
            print(f"使用 Hugging Face 模型: {args.model_path}")

    if not os.path.exists(args.dev_path):
        print(f"错误：开发集路径不存在 - {args.dev_path}")
        return
    if not os.path.exists(args.tables_path):
        print(f"错误：表格信息路径不存在 - {args.tables_path}")
        return
    if not os.path.exists(args.template_path):
        print(f"错误：模板路径不存在 - {args.template_path}")
        return
    if not os.path.exists(args.db_root_path):
        print(f"错误：数据库路径不存在 - {args.db_root_path}")
        return

    if is_main_process:
        print("正在加载数据...")
    dev_data, schema_map = load_data(args.dev_path, args.tables_path)
    if is_main_process:
        print(f"加载了 {len(dev_data)} 条开发集数据")

    if is_main_process:
        print("正在加载 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(actual_model_path, trust_remote_code=True)
    except ValueError:
        if is_main_process:
            print("Fast tokenizer 加载失败，回退到 slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(actual_model_path, trust_remote_code=True, use_fast=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process:
        print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        actual_model_path,
        torch_dtype=resolve_dtype(args.torch_dtype),
        device_map=None if args.use_accelerate_fsdp else "auto",
        trust_remote_code=True,
    )
    if args.use_accelerate_fsdp:
        model = accelerator.prepare(model)

    model.eval()
    if is_main_process:
        print("模型加载完成！")

    system_prompt = load_system_prompt(args.system_prompt_path)

    if is_main_process:
        print("正在准备 prompts...")
    if "OmniSQL-" in actual_model_path:
        if is_main_process:
            print("Using Omni format prompts...")
        tasks = prepare_prompts_omni(dev_data, schema_map, args.template_path, tokenizer, system_prompt)
    else:
        tasks = prepare_prompts(dev_data, schema_map, args.template_path, tokenizer, system_prompt)
    if is_main_process:
        print(f"准备了 {len(tasks)} 个推理任务")

    if args.strategy == "greedy":
        do_sample = False
        num_return_sequences = 1
        temperature = 1.0
        top_p = 0.95
        if is_main_process:
            print("贪心策略: do_sample=False, n=1")
    else:
        do_sample = True
        num_return_sequences = args.num_candidates
        temperature = 0.7 if args.temperature == 0.0 else args.temperature
        top_p = args.top_p
        if is_main_process:
            print(f"投票策略: do_sample=True, temperature={temperature}, n={num_return_sequences}")

    if accelerator is None:
        local_tasks = tasks
    else:
        rank = accelerator.process_index
        world = accelerator.num_processes
        local_tasks = tasks[rank::world]
        if is_main_process:
            print(f"FSDP task sharding: total={len(tasks)}, world={world}, local_rank0={len(local_tasks)}")

    if is_main_process:
        print("正在进行分批推理...")

    local_results = []
    for batch_start in range(0, len(local_tasks), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(local_tasks))
        batch_prompts = [local_tasks[i]["prompt"] for i in range(batch_start, batch_end)]
        if is_main_process:
            print(f"正在处理本进程第 {batch_start + 1} - {batch_end} 个样本...")

        batch_outputs = batched_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_new_tokens=args.max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            device=device,
        )
        for task, responses in zip(local_tasks[batch_start:batch_end], batch_outputs):
            sqls = [extract_sql_from_output(response) for response in responses]
            item = task["original_item"]
            item["responses"] = responses
            item["pred_sqls"] = sqls
            qid = int(item.get("question_id", -1))
            local_results.append({"question_id": qid, "item": item})

        if is_main_process:
            print(f"\n{'=' * 60}")
            print(f"本进程第 {batch_end} 个样本的生成内容:")
            if batch_outputs:
                sample_idx = batch_start + len(batch_outputs) - 1
                sample_task = local_tasks[sample_idx]
                sample_text = batch_outputs[-1][0] if batch_outputs[-1] else ""
                print(f"问题: {sample_task['original_item'].get('question', '')}")
                print(f"数据库: {sample_task['original_item'].get('db_id', '')}")
                print(f"生成内容: {sample_text}")
            print(f"{'=' * 60}\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if accelerator is None:
        merged = local_results
    else:
        accelerator.wait_for_everyone()
        gathered = gather_object(local_results)
        merged = []
        for part in gathered:
            merged.extend(part)

    if not is_main_process:
        return

    merged.sort(key=lambda x: x.get("question_id", -1))
    results = [x["item"] for x in merged]

    model_name = args.model_path.rstrip("/").split("/")[-1]
    output_path_plain = args.output_name + ".json"
    output_path_with_model = args.output_name + "_" + model_name + ".json"

    serialized = json.dumps(results, indent=2, ensure_ascii=False)
    with open(output_path_plain, "w", encoding="utf-8") as f:
        f.write(serialized)
    with open(output_path_with_model, "w", encoding="utf-8") as f:
        f.write(serialized)

    print(f"推理完成，结果已保存: {output_path_plain}")
    print(f"推理完成，结果已保存: {output_path_with_model}")


if __name__ == "__main__":
    main()
