import json
import argparse
import os
from tqdm import tqdm

SCM_PROMPT_TEMPLATE = """
As an expert Data Scientist specializing in training Structural Causal Models (SCMs) for Text-to-SQL tasks, your mission is to perform a STEP-BY-STEP evaluation of a Chain-of-Thought (CoT). Your goal is to analyze the DYNAMICS of the reasoning process by assessing how each individual step contributes to revealing and locking in the final SQL's causal intents.
Your task is to:
1.  First, parse the CoT into a sequence of distinct, logical reasoning steps.
2.  Then, for EACH step, provide a structured evaluation based on the **strict criteria** below. Be critical and differentiate your scores.

**Step-wise Evaluation Criteria:**

* `description`: Briefly summarize what this step accomplishes.
* `focused_intent`: Identify the primary causal intent(s) this step addresses. Choose from: `["SELECT", "FROM", "JOIN", "WHERE", "GROUP_BY", "HAVING", "ORDER_BY", "LIMIT", "AGGREGATE", "PLANNING", "UNKNOWN"]`.

* **`convergence_score` (1-10): Focus on IMPACT.** How much does this step **reduce uncertainty** and definitively lock in a specific part of the final query plan (the `focused_intent`)?
    * **A high score (9-10)** means this step makes a KEY decision, leaving little ambiguity for its focused intent (e.g., explicitly choosing `COUNT(*)` over `SUM()`, or identifying a complex `JOIN` condition).
    * **A low score (1-4)** means the step is very general or preparatory, stating a goal without making a concrete decision (e.g., "I need to look at the tables first."). This step might be efficient but has a low impact on convergence.

* **`efficiency_score` (1-10): Focus on NECESSITY & CONCISENESS.** Is this step a necessary and concise part of the most direct logical path to the solution?
    * **A high score (9-10)** means the step is essential and not redundant.
    * **A low score (1-4)** indicates the step is redundant (e.g., repeating information), explores an irrelevant dead-end, or could have been combined with another step.

**--- CRITICAL DISTINCTION FOR SCORING ---**
A step like *"First, I need to figure out the correct filter conditions."* can be very **efficient** (score 9-10, as it's a necessary planning step) but has very low **convergence** (score 2-3, as it has not decided *what* the conditions are yet).
Conversely, a step that correctly performs a complex calculation for a dead-end path is **inefficient** (score 1-4) even if it locally converges on a result.
**Do not assign high scores to both metrics by default.**


**Output Format:**
You MUST return your evaluation as a single, valid JSON object, without any additional explanatory text or Markdown formatting. The JSON structure should be as follows:
{{
  "step_wise_evaluation": [
    {{
      "step": 1,
      "description": "A brief summary of what this step accomplishes.",
      "focused_intent": ["<INTENT_1>", "<INTENT_2>"],
      "convergence_score": <score_int>,
      "efficiency_score": <score_int>
    }},
    {{
      "step": 2,
      "description": "...",
      "focused_intent": ["..."],
      "convergence_score": <score_int>,
      "efficiency_score": <score_int>
    }},
    ...
  ],
  "overall_justification": "A summary of the CoT's reasoning flow quality from an SCM perspective, highlighting its strengths and weaknesses in progressively reducing uncertainty.",
  "final_parsed_intents": {{
    "select": ["<column1>"],
    "from": ["<table1>"],
    "join": [{{"table": "<table2>", "on": "<condition>"}}],
    "where": [{{"column": "<col>", "op": "<op>", "value": "<val>"}}],
    "group_by": ["<column>"],
    "order_by": {{"column": "<col>", "direction": "<asc/desc>"}},
    "aggregate": [{{"function": "<func>", "column": "<col>"}}]
  }}
}}

---
**Please evaluate the following content:**

**[User Question]**
{question}

**[Final SQL]**
```sql
{sql}
```
**[Chain-of-Thought (CoT)]**
{cot}
"""


def convert_to_batch_format(input_path, output_path, model_name="gemini-2.5-pro",max_sample=None):
    """
    将训练数据转换为Gemini Batch API格式

    Args:
        input_path: 输入的train.json文件路径
        output_path: 输出的batch文件路径
        model_name: 使用的模型名称
    """

    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        return

    # 读取原始数据
    print(f"📖 读取数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"📊 数据总量: {len(dataset)}")
    if max_sample is not None:
        dataset = dataset[3000:3000+max_sample]
        print(f"🔪 截断为前 {max_sample} 条数据")
    # 转换为batch格式
    batch_data = []

    for idx, entry in enumerate(tqdm(dataset, desc="转换数据格式")):
        # 构建提示文本
        hint = entry.get('external_knowledge', '')
        question = hint + " " + entry["question"]

        prompt_text = SCM_PROMPT_TEMPLATE.format(
            question=question,
            sql=entry["sql"],
            cot=entry["cot"]
        )

        # 构建batch请求格式
        batch_request = {
            "key": f"request-{idx + 1}",
            "request": {
                "contents": [{
                    "parts": [{
                        "text": prompt_text
                    }],
                    'role': 'user'
                }],
            },
            # 保存原始数据的元信息，方便后续处理
            "metadata": {
                "db_id":entry['db_id'],
                "original_question": entry["question"],
                "original_sql": entry["sql"],
                "original_cot": entry["cot"],
                "external_knowledge": entry.get("external_knowledge", "")
            }
        }

        batch_data.append(batch_request)

    # 保存为JSONL格式 (每行一个JSON对象)
    print(f"💾 保存batch文件: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for request in batch_data:
            # 移除metadata，只保留Gemini API需要的部分
            batch_line = {
                "key": request["key"],
                "request": request["request"]
            }
            f.write(json.dumps(batch_line, ensure_ascii=False) + '\n')

    # 单独保存metadata文件，用于后续结果合并
    metadata_path = output_path.replace('.jsonl', '_metadata.json')
    print(f"💾 保存元数据文件: {metadata_path}")
    metadata = {req["key"]: req["metadata"] for req in batch_data}
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成!")
    print(f"   📄 Batch文件: {output_path} ({len(batch_data)} 个请求)")
    print(f"   📄 元数据文件: {metadata_path}")
    print(f"\n📋 使用方法:")
    print(f"   1. 上传batch文件到Gemini API")
    print(f"   2. 获取结果后使用元数据文件合并原始数据")


def merge_batch_results(batch_results_path, metadata_path, output_path):
    """
    合并batch结果和原始数据

    Args:
        batch_results_path: Gemini batch API返回的结果文件
        metadata_path: 元数据文件
        output_path: 最终输出文件
    """
    print(f"📖 读取batch结果: {batch_results_path}")
    batch_results = {}
    with open(batch_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            batch_results[result["key"]] = result

    print(f"📖 读取元数据: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 合并数据
    merged_data = []
    for key, meta in tqdm(metadata.items(), desc="合并结果"):
        if key in batch_results:
            result = batch_results[key]

            # 解析评分结果
            try:
                if "response" in result and "candidates" in result["response"]:
                    content = result["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    cleaned_content = content.strip()
                    if cleaned_content.startswith("```json"):
                        cleaned_content = cleaned_content[7:]
                    if cleaned_content.endswith("```"):
                        cleaned_content = cleaned_content[:-3]

                    scm_evaluation = json.loads(cleaned_content.strip())
                else:
                    scm_evaluation = {"error": "No valid response"}
            except Exception as e:
                scm_evaluation = {"error": f"Failed to parse: {str(e)}"}

            # 重构原始数据格式
            combined_entry = {
                "question": meta["original_question"],
                "sql": meta["original_sql"],
                "cot": meta["original_cot"],
                "external_knowledge": meta["external_knowledge"],
                "scm_evaluation": scm_evaluation
            }

            merged_data.append(combined_entry)
        else:
            print(f"⚠️  警告: 找不到 {key} 的结果")

    # 保存合并结果
    print(f"💾 保存合并结果: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in merged_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✅ 合并完成! 共处理 {len(merged_data)} 条记录")


def main():
    parser = argparse.ArgumentParser(description="转换数据为Gemini Batch API格式")
    parser.add_argument("--action", type=str, choices=["convert", "merge"], required=True,
                        help="操作类型: convert(转换) 或 merge(合并结果)")
    parser.add_argument("--input_path", type=str,
                        default="/home/banlujie/data/synsql_5k/train.json",
                        help="输入文件路径")
    parser.add_argument("--output_path", type=str,
                        default="/home/banlujie/data/synsql_5k/batch_requests.jsonl",
                        help="输出文件路径")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-pro",
                        help="使用的模型名称")
    parser.add_argument("--batch_results_path", type=str,
                        help="batch结果文件路径(merge时使用)")
    parser.add_argument("--metadata_path", type=str,
                        help="元数据文件路径(merge时使用)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大转换样本数，默认转换全部")
    args = parser.parse_args()

    if args.action == "convert":
        convert_to_batch_format(
            args.input_path, args.output_path, args.model_name, args.max_samples)
    elif args.action == "merge":
        if not args.batch_results_path or not args.metadata_path:
            print("❌ merge操作需要指定 --batch_results_path 和 --metadata_path")
            return
        merge_batch_results(args.batch_results_path,
                            args.metadata_path, args.output_path)


if __name__ == "__main__":
    main()
