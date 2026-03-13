from tqdm import tqdm
import time
import argparse
from google.genai import types
from google import genai
import json
import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
try:
    api_key = os.environ["GOOGLE_API_KEY"]
    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }
    client = genai.Client()
except KeyError:
    print("Error: API Key is not configured!\n")
    exit()

SCM_PROMPT_TEMPLATE = """

I will provide you with the **User Question**, the **Final SQL**, and the **Chain-of-Thought (CoT)**.

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


def batch_scoring(input_file: str, output_file: str, model_name: str = "gemini-2.5-pro", batch_name=None):
    if batch_name is not None:
        completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

        print(f"Polling status for job: {batch_name}")
        batch_job = client.batches.get(name=batch_name)  # Initial get
        while batch_job.state.name not in completed_states:
            print(f"Current state: {batch_job.state.name}")
            time.sleep(30)  # Wait for 30 seconds before polling again
            batch_job = client.batches.get(name=batch_name)

        print(f"Job finished with state: {batch_job.state.name}")
        if batch_job.state.name == 'JOB_STATE_FAILED':
            print(f"Error: {batch_job.error}")
        elif batch_job.state.name == 'JOB_STATE_SUCCEEDED':

            if batch_job.dest and batch_job.dest.file_name:
                # Results are in a file
                result_file_name = batch_job.dest.file_name
                print(f"Results are in file: {result_file_name}")

                print("Downloading result file content...")
                file_content = client.files.download(file=result_file_name)
                # Process file_content (bytes) as needed
                with open(output_file, 'wb') as f_out:
                    f_out.write(file_content)
            # If batch job was created with inline request
            # (for embeddings, use batch_job.dest.inlined_embed_content_responses)
            elif batch_job.dest and batch_job.dest.inlined_responses:
                # Results are inline
                print("Results are inline:")
                for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                    print(f"Response {i + 1}:")
                    if inline_response.response:
                        # Accessing response, structure may vary.
                        try:
                            print(inline_response.response.text)
                        except AttributeError:
                            print(inline_response.response)  # Fallback
                    elif inline_response.error:
                        print(f"Error: {inline_response.error}")
            else:
                print("No results found (neither file nor inline).")
        else:
            print(f"Job did not succeed. Final state: {batch_job.state.name}")
            if batch_job.error:
                print(f"Error: {batch_job.error}")
    else:
        uploaded_file = client.files.upload(
            file=input_file,
            config=types.UploadFileConfig(
                display_name='cot-batch-scoring', mime_type='jsonl')
        )
        file_batch_job = client.batches.create(
            model=model_name,
            src=uploaded_file.name,
            config={
                'display_name': "soc-scoring-job-1",
            },
        )

        print(f"Created batch job: {file_batch_job.name}")
        job_name = file_batch_job.name
        print(f"Polling status for job: {job_name}")
        batch_job = client.batches.get(name=job_name)

        completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

        print(f"Polling status for job: {job_name}")
        batch_job = client.batches.get(name=job_name)  # Initial get
        while batch_job.state.name not in completed_states:
            print(f"Current state: {batch_job.state.name}")
            time.sleep(30)  # Wait for 30 seconds before polling again
            batch_job = client.batches.get(name=job_name)

        print(f"Job finished with state: {batch_job.state.name}")
        if batch_job.state.name == 'JOB_STATE_FAILED':
            print(f"Error: {batch_job.error}")
        elif batch_job.state.name == 'JOB_STATE_SUCCEEDED':

            if batch_job.dest and batch_job.dest.file_name:
                # Results are in a file
                result_file_name = batch_job.dest.file_name
                print(f"Results are in file: {result_file_name}")

                print("Downloading result file content...")
                file_content = client.files.download(file=result_file_name)
                # Process file_content (bytes) as needed
                with open(output_file, 'wb') as f_out:
                    f_out.write(file_content)
                print(f"Results written to {output_file}")
            # If batch job was created with inline request
            # (for embeddings, use batch_job.dest.inlined_embed_content_responses)
            elif batch_job.dest and batch_job.dest.inlined_responses:
                # Results are inline
                print("Results are inline:")
                for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                    print(f"Response {i + 1}:")
                    if inline_response.response:
                        # Accessing response, structure may vary.
                        try:
                            print(inline_response.response.text)
                        except AttributeError:
                            print(inline_response.response)  # Fallback
                    elif inline_response.error:
                        print(f"Error: {inline_response.error}")
            else:
                print("No results found (neither file nor inline).")
        else:
            print(f"Job did not succeed. Final state: {batch_job.state.name}")
            if batch_job.error:
                print(f"Error: {batch_job.error}")


def download_job_results(job_name: str, output_file: str):
    print(f"Downloading results for job: {job_name}")
    batch_job = client.batches.get(name=job_name)

    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"Job is not in a succeeded state. Current state: {batch_job.state.name}")
        return

    if batch_job.dest and batch_job.dest.file_name:
        result_file_name = batch_job.dest.file_name
        print(f"Results are in file: {result_file_name}")

        print("Downloading result file content...")
        file_content = client.files.download(file=result_file_name)
        with open(output_file, 'wb') as f_out:
            f_out.write(file_content)
        print(f"Results written to {output_file}")
    else:
        print("No result file found in the batch job.")


def score_cot_for_scm_training(data_entry: dict, model_name: str = "gemini-2.5-pro"):
    hint = data_entry['external_knowledge']
    question = hint + " " + data_entry["question"]
    prompt = SCM_PROMPT_TEMPLATE.format(
        question=question,
        sql=data_entry["sql"],
        cot=data_entry["cot"]
    )

    print("正在向 Gemini API 發送 SCM 評分請求...")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction='As a Data Scientist specializing in training Structural Causal Models (SCMs) for Text-to-SQL tasks, your core mission is to evaluate the quality of a Chain-of-Thought (CoT) as training data for an SCM.The focus is not on human-friendliness, but on how clearly and structurally the CoT reveals the "Latent Causal Intents"(e.g., SELECT, WHERE, GROUP BY, AGGREGATE) required to translate natural language into SQL. A high-quality CoT should allow an automated script to easily parse and extract these structured intents.',
                response_mime_type='application/json',
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"調用 API 時發生錯誤: {e}")
        return None


def main(args):
    # download_job_results(job_name="batches/vedfpx2y825z37om49cttaa36g9aowby0r02", output_file=args.output_file)
    # batch_scoring(
    #     input_file=args.cot_path,
    #     output_file=args.output_file,
    #     model_name=args.model_name,
    # batch_name=args.batch_name)
    response = {'step_wise_evaluation': [{'step': 1,
                                          'description': 'Identifies the relevant tables (`packages`, `dependencies`) and the key (`package_id`) that links them.',
                                          'focused_intent': ['FROM', 'JOIN'], 'convergence_score': 7,
                                          'efficiency_score': 9}, {'step': 2,
                                                                   'description': "Translates the user's metaphorical request into concrete output requirements: a list of package names and their associated dependencies as a single comma-separated string.",
                                                                   'focused_intent': ['SELECT', 'AGGREGATE'],
                                                                   'convergence_score': 9, 'efficiency_score': 10},
                                         {'step': 3,
                                          'description': 'Maps the identified requirements to specific SQL clauses (`SELECT`, `FROM`, `JOIN`, `GROUP BY`) and explicitly names the `GROUP_CONCAT` function.',
                                          'focused_intent': ['PLANNING', 'GROUP_BY', 'AGGREGATE'],
                                          'convergence_score': 9, 'efficiency_score': 7}, {'step': 4,
                                                                                           'description': 'Makes the final decision on the JOIN strategy, choosing `LEFT JOIN` to ensure all packages are included, even those without dependencies.',
                                                                                           'focused_intent': ['JOIN'],
                                                                                           'convergence_score': 10,
                                                                                           'efficiency_score': 9}],
                'overall_justification': 'The CoT demonstrates a highly structured and logical reasoning flow, making it excellent for SCM training. It systematically reduces uncertainty by first identifying data sources (FROM/JOIN), then clarifying the complex output format (SELECT/AGGREGATE), planning the query structure, and finally refining the specific join logic. Each step converges on a distinct part of the query with minimal redundancy. The clear separation of concerns makes the causal path from question to SQL components easily parsable.',
                'final_parsed_intents': {'select': ['p.package_name'], 'from': ['packages'],
                                         'join': [{'table': 'dependencies', 'on': 'p.package_id = d.package_id'}],
                                         'where': [], 'group_by': ['p.package_name'], 'order_by': None,
                                         'aggregate': [{'function': 'GROUP_CONCAT', 'column': 'd.dep_package_name'}]}}

    with open("/home/banlujie/data/synsql_5k/scm_cot_batch11.json", "w") as outfile:
        # Iterate over each input file path provided
        input_path = "/home/banlujie/data/synsql_5k/scm_cot_batch1.json"
        print(f"Processing file: {input_path}")
        with open(input_path, "r") as infile:
            for line in infile:
                try:
                    data = json.loads(line)
                    if 'response' in data and 'candidates' in data['response']:
                        for candidate in data['response']['candidates']:
                            for part in candidate['content']['parts']:
                                text_content = part['text']
                                # Clean the string by removing markdown fences and whitespace
                                cleaned_text = text_content.strip()
                                if cleaned_text.startswith("```json"):
                                    cleaned_text = cleaned_text[7:]
                                if cleaned_text.endswith("```"):
                                    cleaned_text = cleaned_text[:-3]

                                # Parse the cleaned string into a JSON object
                                inner_json_obj = json.loads(cleaned_text.strip())

                                # Write the JSON object as a string to the output file
                                outfile.write(json.dumps(inner_json_obj) + '\n')
                except json.JSONDecodeError as e:
                    print(f"Could not process line in {input_path}: {line.strip()} - Error: {e}")
            outfile.write(json.dumps(response) + '\n')

    # train_sample = data[1000]
    #
    # training_data_sample = {
    #     "db_id": train_sample["db_id"],
    #     "sql_complexity": "Complex",
    #     "question_style": train_sample["question_style"],
    #     "question": train_sample['question'],
    #     "external_knowledge": train_sample['external_knowledge'],
    #     "cot": train_sample['cot'],
    #     "sql": train_sample['sql']
    # }


    # response = score_cot_for_scm_training(data_entry=training_data_sample, model_name=args.model_name)
    # print(response)
    # print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于 vLLM 的 SQL 生成评估脚本")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-pro",
                        help="evaluation model")
    parser.add_argument("--cot_path", type=str, default="/home/banlujie/data/synsql_5k/batch_requests.jsonl",
                        help="CoT dataset path")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="生成温度")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="最大生成token数")
    parser.add_argument("--output_file", type=str, default="/home/banlujie/data/synsql_5k/scm_cot_batch.json",
                        help="推理结果输出文件路径")
    parser.add_argument("--batch_name", type=str, default=None,
                        help="existing batch job name to poll results for")
    args = parser.parse_args()
    main(args)
