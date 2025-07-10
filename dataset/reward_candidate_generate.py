import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


class CandidateGenerator:
    def __init__(self, sft_model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(sft_model_path).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_for_prompt(self, prompt: str, num_candidates: int = 4) -> list[str]:
        """為單一prompt生成多個不同的SQL候選答案。"""
        candidates = []
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 策略2: 多樣性採樣 (Sampling with Temperature)
        with torch.no_grad():
            # 使用 do_sample=True 並調整 temperature 和 top_p
            sampled_outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                num_return_sequences=num_candidates - 1,  # 生成剩餘的候選者
            )
            for output in sampled_outputs:
                sql = self.tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                candidates.append(sql.strip())

        # 去除重複的生成結果
        return list(set(candidates))


# --- 使用範例 ---
if __name__ == '__main__':
    SFT_MODEL_PATH = "../checkpoint/phi4_mini_reasoning"  # 換成您自己的模型路徑
    generator = CandidateGenerator(sft_model_path=SFT_MODEL_PATH)

    # 假設這是從BIRD數據集讀取的一條數據
    prompt_example = "Question: Show the names of all users older than 25.\nSchema: CREATE TABLE users (id INT, name VARCHAR, age INT)\nSQL: SELECT * FROM users WHERE age > 25;"

    generated_sqls = generator.generate_for_prompt(prompt_example, num_candidates=4)

    print(f"--- Generated Candidates for Prompt ---")
    print(f"Prompt: {prompt_example}")
    print("-" * 20)
    for i, sql in enumerate(generated_sqls):
        print(f"Candidate {i + 1}: {sql}")