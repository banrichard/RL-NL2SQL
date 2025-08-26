import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os


class CreditDataGeneratorFromRM:
    """
    使用訓練好的Reward Model，通過反事實分析為CreditAssignmentModel生成訓練數據。
    """

    def __init__(self, sft_model_path: str, reward_model_path: str):
        print("Initializing models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 載入SFT Actor和其分詞器 (用於生成SQL)
        self.actor_tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        self.actor_model = AutoModelForCausalLM.from_pretrained(sft_model_path).to(self.device)
        self.actor_model.eval()

        # 2. 載入訓練好的Reward Model和其分詞器 (用於快速打分)
        # 假設RM是基於一個序列分類架構訓練的
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path).to(self.device)
        self.reward_model.eval()

        # 統一padding token
        if self.actor_tokenizer.pad_token is None:
            self.actor_tokenizer.pad_token = self.actor_tokenizer.eos_token
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token

        # 獲取用於遮蔽的token id
        self.mask_token_id = self.actor_tokenizer.mask_token_id if self.actor_tokenizer.mask_token_id is not None else self.actor_tokenizer.unk_token_id
        print(f"Using Token ID {self.mask_token_id} for masking.")
        print("Initialization complete. Models are on device:", self.device)

    def get_reward_score(self, prompt: str, sql: str) -> float:
        """使用Reward Model為 (prompt, sql) 對打分。"""
        # RM的輸入通常是 prompt 和 sql 的拼接
        text = prompt + " " + sql
        inputs = self.reward_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            # 假設RM輸出一個單一的分數
            score = outputs.logits.squeeze().item()
        return score

    def generate_dataset(self, source_dataset: list, output_path: str, quality_threshold: float = 0.5):
        """
        主流程：生成並儲存Credit Model的數據集。

        Args:
            source_dataset (list): 包含 {'question': str, 'schema': str} 的列表。
            output_path (str): 數據集儲存路徑。
            quality_threshold (float): RM對原始生成SQL的最低分數要求，低於此分數的樣本將被跳過。
        """
        credit_dataset = []
        print(f"Starting dataset generation for {len(source_dataset)} source prompts...")

        for sample in tqdm(source_dataset, desc="Generating Credit Data"):
            # 為了簡化，這裡將schema直接拼接到prompt中
            prompt = f"Schema: {sample['schema']}\nQuestion: {sample['question']}\nSQL:"

            # 步驟一：讓SFT Actor生成一個高質量的SQL
            inputs = self.actor_tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.actor_model.generate(
                    **inputs, max_new_tokens=128, do_sample=False, num_beams=1
                )
            generated_sql_ids = generated_ids[0, inputs['input_ids'].shape[1]:]
            generated_sql_str = self.actor_tokenizer.decode(generated_sql_ids, skip_special_tokens=True).strip()

            # 步驟二：用RM計算基準總獎勵 R_total
            R_total = self.get_reward_score(prompt, generated_sql_str)

            # 質量控制：如果RM認為原始生成的SQL質量太差，就跳過這個樣本
            if R_total < quality_threshold:
                continue

            # 步驟三：反事實分析循環
            for i in range(len(generated_sql_ids)):
                # a. 構造上下文前綴
                context_prefix_ids = generated_sql_ids[:i]
                context_prefix_str = self.actor_tokenizer.decode(context_prefix_ids, skip_special_tokens=True).strip()

                # b. 構造反事實SQL (遮蔽當前token)
                counterfactual_ids = torch.cat([
                    context_prefix_ids,
                    torch.tensor([self.mask_token_id], device=self.device, dtype=torch.long),
                    generated_sql_ids[i + 1:]
                ])
                counterfactual_sql_str = self.actor_tokenizer.decode(counterfactual_ids,
                                                                     skip_special_tokens=True).strip()

                # c. 用RM計算反事實獎勵
                R_counterfactual = self.get_reward_score(prompt, counterfactual_sql_str)

                # d. 計算信用值
                credit_value = R_total - R_counterfactual

                # e. 儲存訓練樣本
                credit_dataset.append({
                    "context_prefix": context_prefix_str,
                    "credit_value": credit_value
                })

        # 步驟四：保存數據集
        print(f"\nGenerated {len(credit_dataset)} training samples. Saving to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in credit_dataset:
                f.write(json.dumps(entry) + '\n')

        print("Dataset generation complete.")


if __name__ == '__main__':
    # --- 配置 (請替換為您自己的真實路徑) ---
    SFT_MODEL_PATH = "path/to/your/sft_phi4_model"
    REWARD_MODEL_PATH = "path/to/your/trained_reward_model"
    OUTPUT_FILE = "credit_assignment_data.jsonl"

    # --- 模擬的源數據集 ---
    SOURCE_DATA = [
        {"schema": "CREATE TABLE users (id INT, name VARCHAR)", "question": "Show all user names."},
        {"schema": "CREATE TABLE products (pid INT, stock INT)", "question": "How many products are out of stock?"}
        # ... 在實際中，這裡會有成千上萬條來自BIRD等數據集的數據
    ]

    # 執行 (取消以下註解以運行)
    # try:
    #     generator = CreditDataGeneratorFromRM(
    #         sft_model_path=SFT_MODEL_PATH,
    #         reward_model_path=REWARD_MODEL_PATH
    #     )
    #     generator.generate_dataset(source_dataset=SOURCE_DATA, output_path=OUTPUT_FILE)
    # except Exception as e:
    #     print(f"An error occurred. Make sure your model paths are correct. Error: {e}")
    #     print("This script requires valid model paths to run.")

    print("Script structure is ready. Please update model paths and uncomment the main execution block to run.")