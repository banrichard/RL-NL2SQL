#!/usr/bin/env python3
"""
基于 vLLM 的 SQL 生成评估脚本
用于评估模型在开发集上的性能
"""

import json
import os
import re
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
from collections import defaultdict

# 导入数据集工具函数
from dataset.utils import serialize_schema, execute_sql_with_timeout


def extract_sql_from_output(output_text):
    """
    从模型输出中提取 SQL 查询
    支持多种格式的 SQL 代码块
    """
    # 尝试匹配 <answer>```sql ... ``` </answer> 格式
    answer_pattern = r'<answer>\s*```sql\s*(.*?)\s*```\s*</answer>'
    match = re.search(answer_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 尝试匹配普通的 ```sql ... ``` 格式
    sql_pattern = r'```sql\s*(.*?)\s*```'
    match = re.search(sql_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 尝试匹配 ``` ... ``` 格式（无 sql 标识）
    code_pattern = r'```\s*(.*?)\s*```'
    match = re.search(code_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        sql_candidate = match.group(1).strip()
        # 简单检查是否像 SQL（包含 SELECT、INSERT、UPDATE、DELETE 等关键词）
        if any(keyword in sql_candidate.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            return sql_candidate
    
    return None


def evaluate_sql_execution(predicted_sql, ground_truth_sql, db_path, timeout=10):
    """
    评估 SQL 执行结果是否正确
    """
    try:
        # 执行预测的 SQL
        pred_result = execute_sql_with_timeout(db_path, predicted_sql, timeout)
        if isinstance(pred_result, str) and pred_result.startswith("Error"):
            return {"execution_error": True, "correct": False, "error_msg": pred_result}
        
        # 执行标准答案 SQL
        gt_result = execute_sql_with_timeout(db_path, ground_truth_sql, timeout)
        if isinstance(gt_result, str) and gt_result.startswith("Error"):
            return {"execution_error": True, "correct": False, "error_msg": f"Ground truth error: {gt_result}"}
        
        # 比较结果
        is_correct = pred_result == gt_result
        return {
            "execution_error": False,
            "correct": is_correct,
            "pred_result": pred_result,
            "gt_result": gt_result
        }
    
    except Exception as e:
        return {"execution_error": True, "correct": False, "error_msg": str(e)}


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


def prepare_prompts(dev_data, schema_map, template_path, tokenizer):
    """
    准备所有的推理 prompts
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    tasks = []
    for item in dev_data:
        db_id = item.get('db_id')
        if not db_id or db_id not in schema_map:
            print(f"Warning: db_id '{db_id}' not found in schema_map, skipping...")
            continue
        
        schema_str = serialize_schema(schema_map[db_id])
        evidence = item.get('evidence', '')
        question = item['question'] + (" " + evidence if evidence else "")
        
        prompt_text = template.format(
            db_engine="SQLite",
            db_id=db_id,
            db_details=schema_str,
            question=question
        )
        
        messages = [{"role": "user", "content": prompt_text}]
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
    parser.add_argument("--model_path", type=str, default="/home/banlujie/RL-NL2SQL/ckpt/phi4_mini_reasoning",
                        help="模型路径")
    parser.add_argument("--dev_path", type=str, default="/home/banlujie/data/bird_dev/dev.json",
                        help="开发集路径")
    parser.add_argument("--tables_path", type=str, default="/home/banlujie/data/bird_dev/dev_tables.json",
                        help="表格信息路径")
    parser.add_argument("--db_root_path", type=str, default="/home/banlujie/data/bird_dev/dev_databases",
                        help="数据库文件根目录")
    parser.add_argument("--template_path", type=str, default="/home/banlujie/RL-NL2SQL/templates/sql_generation.txt",
                        help="prompt 模板路径")
    parser.add_argument("--output_path", type=str, default="/home/banlujie/RL-NL2SQL/eval_results.json",
                        help="评估结果输出路径")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="生成温度")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="最大生成token数")
    parser.add_argument("--timeout", type=int, default=10,
                        help="SQL执行超时时间（秒）")
    
    args = parser.parse_args()
    
    print("=== 基于 vLLM 的 SQL 生成评估 ===")
    print(f"模型路径: {args.model_path}")
    print(f"开发集路径: {args.dev_path}")
    print(f"数据库路径: {args.db_root_path}")
    
    # 检查路径是否存在
    if not os.path.exists(args.model_path):
        print(f"错误：模型路径不存在 - {args.model_path}")
        return
    
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
    llm = LLM(model=args.model_path, tensor_parallel_size=torch.cuda.device_count())
    print("模型加载完成！")
    
    # 准备 prompts
    print("正在准备 prompts...")
    tasks = prepare_prompts(dev_data, schema_map, args.template_path, tokenizer)
    print(f"准备了 {len(tasks)} 个推理任务")
    
    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        stop=["```", "</answer>"]
    )
    
    # 批量推理
    print("正在进行批量推理...")
    all_prompts = [task['prompt'] for task in tasks]
    outputs = llm.generate(all_prompts, sampling_params)
    print("推理完成！")
    
    # 评估结果
    print("正在评估结果...")
    results = []
    execution_correct = 0
    sql_extracted = 0
    total_processed = 0
    
    difficulty_stats = defaultdict(lambda: {"total": 0, "correct": 0, "extracted": 0})
    
    for i, output in enumerate(tqdm(outputs, desc="评估进度")):
        original_item = tasks[i]['original_item']
        generated_text = output.outputs[0].text
        
        # 提取 SQL
        predicted_sql = extract_sql_from_output(generated_text)
        
        result = {
            "question_id": original_item.get('question_id', i),
            "db_id": original_item['db_id'],
            "question": original_item['question'],
            "evidence": original_item.get('evidence', ''),
            "ground_truth_sql": original_item['SQL'],
            "predicted_sql": predicted_sql,
            "generated_text": generated_text,
            "difficulty": original_item.get('difficulty', 'unknown')
        }
        
        difficulty = result['difficulty']
        difficulty_stats[difficulty]["total"] += 1
        
        if predicted_sql:
            sql_extracted += 1
            difficulty_stats[difficulty]["extracted"] += 1
            
            # 执行评估
            db_path = os.path.join(args.db_root_path, original_item['db_id'], f"{original_item['db_id']}.sqlite")
            
            if os.path.exists(db_path):
                eval_result = evaluate_sql_execution(
                    predicted_sql, 
                    original_item['SQL'], 
                    db_path, 
                    args.timeout
                )
                result.update(eval_result)
                
                if eval_result['correct']:
                    execution_correct += 1
                    difficulty_stats[difficulty]["correct"] += 1
            else:
                result.update({
                    "execution_error": True,
                    "correct": False,
                    "error_msg": f"Database file not found: {db_path}"
                })
        else:
            result.update({
                "execution_error": True,
                "correct": False,
                "error_msg": "No SQL extracted from output"
            })
        
        results.append(result)
        total_processed += 1
    
    # 计算统计信息
    sql_extraction_rate = sql_extracted / total_processed if total_processed > 0 else 0
    execution_accuracy = execution_correct / total_processed if total_processed > 0 else 0
    
    stats = {
        "total_samples": total_processed,
        "sql_extracted": sql_extracted,
        "execution_correct": execution_correct,
        "sql_extraction_rate": sql_extraction_rate,
        "execution_accuracy": execution_accuracy,
        "difficulty_breakdown": dict(difficulty_stats)
    }
    
    # 保存结果
    output_data = {
        "evaluation_stats": stats,
        "detailed_results": results
    }
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\n=== 评估结果 ===")
    print(f"总样本数: {total_processed}")
    print(f"SQL 提取成功: {sql_extracted} ({sql_extraction_rate:.2%})")
    print(f"执行正确: {execution_correct} ({execution_accuracy:.2%})")
    
    print("\n=== 按难度分类的结果 ===")
    for difficulty, stats_dict in difficulty_stats.items():
        total = stats_dict["total"]
        extracted = stats_dict["extracted"]
        correct = stats_dict["correct"]
        if total > 0:
            print(f"{difficulty}: {correct}/{total} ({correct/total:.2%}) [提取率: {extracted/total:.2%}]")
    
    print(f"\n结果已保存到: {args.output_path}")


if __name__ == "__main__":
    main()
