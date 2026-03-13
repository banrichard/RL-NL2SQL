#!/usr/bin/env python3
"""
基于 vLLM 的 SQL 生成评估脚本
用于评估模型在开发集上的性能
"""

import json
import os
import re
import argparse
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer

# 导入数据集工具函数
from dataset.utils import serialize_schema


def extract_sql_from_output(output_text):
    """
    从模型输出中提取 SQL 查询
    支持多种格式的 SQL 代码块
    """
    pattern = r"```sql\s*(.*?)\s*```"
    sql_blocks = re.findall(pattern, output_text, re.DOTALL)
    if sql_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        return last_sql

    # 1. 优先提取 Qwen XML 格式的 <sql> 标签
    xml_sql_pattern = r'<sql>\s*(.*?)\s*</sql>'
    match = re.search(xml_sql_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. 尝试匹配 <answer>```sql ... ``` </answer> 格式
    answer_pattern = r'<answer>\s*```sql\s*(.*?)\s*```\s*</answer>'
    match = re.search(answer_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 3. 尝试匹配普通的 ```sql ... ``` 格式

    # 4. 尝试匹配 ``` ... ``` 格式（无 sql 标识）
    code_pattern = r'```\s*(.*?)\s*```'
    match = re.search(code_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        sql_candidate = match.group(1).strip()
        # 简单检查是否像 SQL（包含 SELECT、INSERT、UPDATE、DELETE 等关键词）
        if any(keyword in sql_candidate.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            return sql_candidate

    # 5. 直接提取首个 SQL 语句 - 改进版本
    # 匹配从 SELECT/INSERT/UPDATE/DELETE 开始到分号或行末的完整语句
    sql_line_pattern = r'((?:SELECT|INSERT|UPDATE|DELETE)(?:[^;])+(?:;|$))'
    match = re.search(sql_line_pattern, output_text,
                      re.IGNORECASE | re.MULTILINE)
    if match:
        sql_candidate = match.group(1).strip()
        # 确保不是只有一个关键词
        if len(sql_candidate.split()) > 1:
            return sql_candidate

    # 6. 更宽松的匹配 - 匹配多行 SQL
    multiline_sql_pattern = r'((?:SELECT|INSERT|UPDATE|DELETE)(?:.|\n)*?)(?:\n\s*\n|\n\s*$|$)'
    match = re.search(multiline_sql_pattern, output_text, re.IGNORECASE)
    if match:
        sql_candidate = match.group(1).strip()
        # 确保有基本的 SQL 结构
        if len(sql_candidate.split()) > 2 and ('FROM' in sql_candidate.upper() or 'INTO' in sql_candidate.upper()):
            return sql_candidate

    # 7. 按行查找最长的 SQL 语句
    longest_sql = None
    max_length = 0
    for line in output_text.splitlines():
        line = line.strip()
        if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE')):
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
    # 加载开发集数据
    try:
        with open(dev_path, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Failed to load as JSON: {e}")
        print("Trying to load as JSONL format...")
        dev_data = []
        with open(dev_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        dev_data.append(json.loads(line))
                    except json.JSONDecodeError as line_error:
                        print(f"Error parsing line {line_num}: {line_error}")
        print(f"Successfully loaded {len(dev_data)} records from JSONL format")

    # 加载表格信息
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)

    schema_map = {db['db_id']: db for db in tables_data}

    return dev_data, schema_map

def prepare_prompts_omni(dev_data, schema_map, template_path, tokenizer, system_prompt_path=None):
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # 加载自定义 system prompt 或使用默认的
    if system_prompt_path and os.path.exists(system_prompt_path):
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
        print(f"使用自定义 system prompt: {system_prompt_path}")
    else:
        # 默认 system prompt
        system_prompt = """You are an expert SQL developer with deep knowledge of database design and query optimization. 
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
        print("使用默认 system prompt")

    tasks = []
    for item in dev_data:
        db_id = item.get('db_id')
        if not db_id or db_id not in schema_map:
            print(
                f"Warning: db_id '{db_id}' not found in schema_map, skipping...")
            continue

        schema_str = item.get('db_details')
        question = item.get('question')

        prompt_text = template.format(
            db_id=db_id,
            db_details=schema_str,
            question=question
        )

        # 使用 system + user 消息结构
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ]
        final_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        tasks.append({
            "prompt": final_prompt,
            "original_item": item
        })

    return tasks

def prepare_prompts(dev_data, schema_map, template_path, tokenizer, system_prompt_path=None):
    """
    准备所有的推理 prompts
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # 加载自定义 system prompt 或使用默认的
    if system_prompt_path and os.path.exists(system_prompt_path):
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
        print(f"使用自定义 system prompt: {system_prompt_path}")
    else:
        # 默认 system prompt
        system_prompt = """You are an expert SQL developer with deep knowledge of database design and query optimization. 
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
        print("使用默认 system prompt")

    tasks = []
    for item in dev_data:
        db_id = item.get('db_id')
        if not db_id or db_id not in schema_map:
            print(
                f"Warning: db_id '{db_id}' not found in schema_map, skipping...")
            continue

        schema_str = serialize_schema(schema_map[db_id])
        evidence = item.get('evidence', '')
        question = item['question']+evidence

        prompt_text = template.format(
            db_id=db_id,
            db_details=schema_str,
            question=question
        )

        # 使用 system + user 消息结构
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ]
        final_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        tasks.append({
            "prompt": final_prompt,
            "original_item": item
        })

    return tasks


def main():
    parser = argparse.ArgumentParser(description="基于 vLLM 的 SQL 生成评估脚本")
    parser.add_argument("--model_path", type=str, default="/data/banlujie/RL-NL2SQL/ckpt/phi4_mini_reasoning",
                        help="模型路径")
    parser.add_argument("--dev_path", type=str, default="/data/banlujie/data/bird_dev/dev.json",
                        help="开发集路径")
    parser.add_argument("--tables_path", type=str, default="/data/banlujie/data/bird_dev/dev_tables.json",
                        help="表格信息路径")
    parser.add_argument("--db_root_path", type=str, default="/data/banlujie/data/bird_dev/dev_databases",
                        help="数据库文件根目录")
    parser.add_argument("--template_path", type=str, default="/data/banlujie/RL-NL2SQL/templates/sql_generation.txt",
                        help="prompt 模板路径")
    parser.add_argument("--system_prompt_path", type=str, default=None,
                        help="自定义 system prompt 文件路径（可选）")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="生成温度")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="最大生成token数")
    parser.add_argument("--timeout", type=int, default=60,
                        help="SQL执行超时时间（秒）")
    parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy", "voting"],
                        help="生成策略: greedy(贪心) 或 voting(投票)")
    parser.add_argument("--num_candidates", type=int, default=5,
                        help="投票策略时的候选SQL数量")
    parser.add_argument("--gpu_utilization", type=float, default=0.6,
                        help="GPU利用率（0.0-1.0），控制GPU显存使用")
    parser.add_argument("--output_name", type=str, default="/data/banlujie/RL-NL2SQL/eval_res/vllm_inference_result",
                        help="推理结果输出文件路径")
    args = parser.parse_args()

    print("=== 基于 vLLM 的 SQL 生成评估 ===")
    print(f"模型路径: {args.model_path}")
    print(f"开发集路径: {args.dev_path}")
    print(f"数据库路径: {args.db_root_path}")
    print(f"GPU利用率: {args.gpu_utilization}")
    print(f"温度: {args.temperature}")
    print(f"最大tokens: {args.max_tokens}")

    # 检查路径是否存在
    # 如果模型路径包含 "/"，可能是 Hugging Face 模型 ID，不需要检查本地存在性
    if "/" not in args.model_path and not os.path.exists(args.model_path):
        print(f"错误：模型路径不存在 - {args.model_path}")
        return
    elif "/" in args.model_path:
        print(f"使用 Hugging Face 模型: {args.model_path}")
    else:
        print(f"使用本地模型: {args.model_path}")

    if not os.path.exists(args.dev_path):
        print(f"错误：开发集路径不存在 - {args.dev_path}")
        return

    if not os.path.exists(args.db_root_path):
        print(f"错误：数据库路径不存在 - {args.db_root_path}")
        return

    # 加载数据
    print("正在加载数据...")
    dev_data, schema_map = load_data(args.dev_path, args.tables_path)
    print(f"加载了 {len(dev_data)} 条开发集数据")

    # 加载模型
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if "Qwen2.5-" in args.model_path:
        # 151645 is the token id of <|im_end|> (end of turn token in Qwen2.5)
        stop_token_ids = [151645]
    elif "phi-4-" in args.model_path or "Phi-4-" in args.model_path:
        stop_token_ids = [200020]  # phi-4 uses the same tokenizer as GPT-2
    elif "deepseek-coder-" in args.model_path:
        stop_token_ids = [32021]
    elif "DeepSeek-Coder-V2" in args.model_path:
        stop_token_ids = [100001]
    elif "OpenCoder-" in args.model_path:
        stop_token_ids = [96539]
    elif "Meta-Llama-" in args.model_path:
        stop_token_ids = [128009, 128001]
    elif "granite-" in args.model_path:
        # <|end_of_text|> is the end token of granite-3.1 and granite-code
        stop_token_ids = [0]
    elif "starcoder2-" in args.model_path:
        stop_token_ids = [0]  # <|end_of_text|> is the end token of starcoder2
    elif "Codestral-" in args.model_path:
        stop_token_ids = [2]
    elif "Mixtral-" in args.model_path:
        stop_token_ids = [2]
    elif "OmniSQL-" in args.model_path:
        stop_token_ids = [151645]  # OmniSQL uses the same tokenizer as Qwen2.5
    else:
        print("Use Qwen2.5's stop tokens by default.")
        stop_token_ids = [151645]
    llm = LLM(
        dtype="bfloat16",
        model=args.model_path,
        max_model_len=args.max_tokens,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=args.gpu_utilization,
        swap_space=42,
    )
    print("模型加载完成！")

    # 准备 prompts
    print("正在准备 prompts...")
    if "OmniSQL-" in args.model_path:
        print("Using Omni format prompts...")
        tasks = prepare_prompts_omni(
            dev_data, schema_map, args.template_path, tokenizer, args.system_prompt_path)
    else:
        tasks = prepare_prompts(
            dev_data, schema_map, args.template_path, tokenizer, args.system_prompt_path)
    print(f"准备了 {len(tasks)} 个推理任务")

    # 根据策略设置生成参数
    print(f"使用策略: {args.strategy}")
    if args.strategy == "greedy":
        sampling_params = SamplingParams(
            stop_token_ids=stop_token_ids,
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            n=1  # 贪心策略只生成1个候选
        )
        print("贪心策略: temperature=0.0, n=1")
    else:  # voting strategy
        sampling_params = SamplingParams(
            temperature=0.7 if args.temperature == 0.0 else args.temperature,
            stop_token_ids=stop_token_ids,
            top_p=0.95,
            max_tokens=4096,
            n=args.num_candidates  # 投票策略生成多个候选
        )
        print(
            f"投票策略: temperature={sampling_params.temperature}, n={args.num_candidates}")

    # 分批推理，每批500个样本
    print("正在进行分批推理...")
    batch_size = 500
    all_outputs = []

    for batch_start in range(0, len(tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks))
        batch_prompts = [tasks[i]['prompt']
                         for i in range(batch_start, batch_end)]

        print(f"正在处理第 {batch_start + 1} - {batch_end} 个样本...")
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(batch_outputs)

        # 打印当前批次的一些生成内容
        print(f"\n{'=' * 60}")
        print(f"第 {batch_end} 个样本的生成内容:")
        if batch_outputs:
            sample_idx = batch_start + len(batch_outputs) - 1
            sample_task = tasks[sample_idx]
            sample_output = batch_outputs[-1]
            print(f"问题: {sample_task['original_item']['question']}")
            print(f"数据库: {sample_task['original_item']['db_id']}")
            print(f"生成内容: {sample_output.outputs[0].text}")
        print(f"{'=' * 60}\n")

    outputs = all_outputs
    results = []
    for data, output in zip(tasks, outputs):
        responses = [o.text for o in output.outputs]
        sqls = [extract_sql_from_output(response) for response in responses]

        data['original_item']["responses"] = responses
        data['original_item']["pred_sqls"] = sqls
        results.append(data['original_item'])
    model_name = args.model_path.split("/")[-1]
    with open(args.output_name+"_"+model_name + ".json", "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
    print("推理完成！")

if __name__ == "__main__":
    main()