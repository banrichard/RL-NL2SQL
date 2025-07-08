import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os


class CreditDataGenerator:
    """
    通过反事实推理，为CreditAssignmentModel创造训练数据。
    """

    def __init__(self, actor_model_name: str):
        """
        初始化生成器，加载Actor模型和分词器。

        Args:
            actor_model_name (str): 您在BIRD上SFT过的模型路径，或一个Hugging Face模型名称。
        """
        print(f"Loading model and tokenizer from: {actor_model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(actor_model_name)
        self.actor_model = AutoModelForCausalLM.from_pretrained(actor_model_name).to(self.device)
        self.actor_model.eval()

        # 设置padding token（如果模型没有的话）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 获取用于遮蔽的token id
        self.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.unk_token_id
        print(f"Using token ID {self.mask_token_id} for masking.")

    def simplified_reward_function(self, generated_sql: str, gold_sql: str) -> float:
        """
        一个简化的、模拟的奖励函数。
        !!! 警告: 这是一个占位符，请在实际项目中替换为真实的SQL执行逻辑 !!!

        Args:
            generated_sql (str): 模型生成的SQL。
            gold_sql (str): 数据集中的标准答案SQL。

        Returns:
            float: 一个奖励分数。
        """
        # 1. 如果SQL被破坏（例如包含mask token），则为语法错误，给予重罚
        if self.tokenizer.mask_token in generated_sql or self.tokenizer.unk_token in generated_sql:
            return -2.0

        # 2. 模拟执行正确性：检查是否包含关键的SQL关键词
        # 这是一个非常粗糙的模拟
        try:
            # 简单检查一些关键词是否存在
            if "SELECT" in generated_sql.upper() and "FROM" in generated_sql.upper():
                # 更进一步，可以检查是否包含gold_sql中的表名或列名
                return 5.0  # 模拟执行成功且结果正确
            else:
                return -1.0  # 模拟语法正确但逻辑错误
        except Exception:
            return -2.0  # 模拟语法错误

    def generate_data(self, source_dataset: list, output_path: str):
        """
        主函数，遍历源数据，生成并保存信用分配数据集。

        Args:
            source_dataset (list): 模拟BIRD的数据集，每个元素是{'question': str, 'gold_sql': str}。
            output_path (str): 生成的数据集要保存的路径 (e.g., 'credit_data.jsonl')。
        """
        credit_dataset = []
        print(f"Starting data generation. Processing {len(source_dataset)} samples.")

        for sample in tqdm(source_dataset, desc="Generating Credit Data"):
            question = sample['question']
            gold_sql = sample['gold_sql']

            # 步骤1: Actor生成SQL
            prompt = f"Question: {question}\nSQL:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.actor_model.generate(
                    **inputs,
                    max_new_tokens=64,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # 只解码生成的部分
            generated_sql_ids = generated_ids[0, inputs['input_ids'].shape[1]:]
            generated_sql_str = self.tokenizer.decode(generated_sql_ids, skip_special_tokens=True).strip()

            # 步骤2: 计算基准总奖励
            R_total = self.simplified_reward_function(generated_sql_str, gold_sql)

            # 如果基准奖励过低，可以跳过，因为它提供不了太多有用的正面信号
            if R_total < 0:
                continue

            # 步骤3: 反事实循环，为每个Token计算信用值
            for i in range(len(generated_sql_ids)):
                # a. 构造上下文前缀
                context_prefix_ids = generated_sql_ids[:i]
                context_prefix_str = self.tokenizer.decode(context_prefix_ids, skip_special_tokens=True).strip()

                # b. 构造反事实SQL (遮蔽当前token)
                counterfactual_ids = torch.cat([
                    generated_sql_ids[:i],
                    torch.tensor([self.mask_token_id], device=self.device, dtype=torch.long),
                    generated_sql_ids[i + 1:]
                ])
                counterfactual_sql_str = self.tokenizer.decode(counterfactual_ids, skip_special_tokens=True).strip()

                # c. 计算反事实奖励
                R_counterfactual = self.simplified_reward_function(counterfactual_sql_str, gold_sql)

                # d. 计算信用值
                credit_value = R_total - R_counterfactual

                # e. 存储训练样本
                training_sample = {
                    # 注意：为了让模型学习，上下文应该是SQL的前缀，而不是完整的对话
                    "context_prefix": context_prefix_str,
                    "credit_value": credit_value
                }
                credit_dataset.append(training_sample)

        # 步骤4: 保存数据集到文件
        print(f"\nGenerated {len(credit_dataset)} training samples. Saving to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in credit_dataset:
                f.write(json.dumps(entry) + '\n')

        print("Data generation complete.")


if __name__ == '__main__':
    # --- 配置 ---
    # 在实际应用中，这里应该是你SFT过的模型的路径
    # huggingface.co上的模型可能无法很好地生成SQL，这里仅作演示
    ACTOR_MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
    OUTPUT_FILE = "credit_assignment_data.jsonl"

    # --- 模拟的源数据集 (模仿BIRD的格式) ---
    FAKE_BIRD_DATASET = [
        {
            "question": "Show the names of all users.",
            "gold_sql": "SELECT name FROM users"
        },
        {
            "question": "How many products are out of stock?",
            "gold_sql": "SELECT count(*) FROM products WHERE stock = 0"
        },
        # ... 在实际中，这里会有成千上万条数据
    ]

    # --- 执行 ---
    generator = CreditDataGenerator(actor_model_name=ACTOR_MODEL_NAME)
    generator.generate_data(source_dataset=FAKE_BIRD_DATASET, output_path=OUTPUT_FILE)

    # --- 验证输出 ---
    if os.path.exists(OUTPUT_FILE):
        print(f"\n--- Verifying output file '{OUTPUT_FILE}' ---")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # 只打印前5行看看
                    break
                print(line.strip())
