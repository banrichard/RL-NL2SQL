import ray
import torch
from openrlhf.trainer.ray.launcher import BaseModelActor
from algorithm.credit_model import CreditModel


@ray.remote(num_gpus=1)
class CreditModelActor(BaseModelActor):
    """
    Ray Actor for Credit Model - 用于在PPO训练中提供信用分配
    """

    def async_init_model_from_pretrained(self, *args, **kwargs):
        return self.init_model_from_pretrained(*args, **kwargs)

    def init_model_from_pretrained(self, strategy, pretrain_path, num_outputs=8):
        """
        初始化预训练的Credit Model
        """
        self.strategy = strategy
        
        # 加载Credit Model
        self.credit_model = CreditModel.from_pretrained(
            pretrain_path,
            num_outputs=num_outputs,
            use_flash_attention_2=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            load_in_4bit=getattr(strategy.args, 'load_in_4bit', False),
            lora_rank=getattr(strategy.args, 'lora_rank', 0),
            lora_alpha=getattr(strategy.args, 'lora_alpha', 16),
            target_modules=getattr(strategy.args, 'target_modules', None),
            lora_dropout=getattr(strategy.args, 'lora_dropout', 0),
        )
        
        # 设置为评估模式
        self.credit_model.eval()
        
        # 加载tokenizer
        from openrlhf.utils import get_tokenizer
        self.tokenizer = get_tokenizer(
            pretrain_path, 
            self.credit_model.model, 
            "left", 
            strategy, 
            use_fast=not getattr(strategy.args, 'disable_fast_tokenizer', False)
        )
        
        print(f"Credit Model Actor initialized with model from {pretrain_path}")
        return self

    def get_credit_scores(self, queries):
        """
        为查询序列计算信用分数
        
        Args:
            queries: List of complete sequences (prompt + response)
            
        Returns:
            List of credit score vectors
        """
        if not hasattr(self, 'credit_model'):
            raise RuntimeError("Credit model not initialized")
            
        credit_scores = []
        
        with torch.no_grad():
            for query in queries:
                # Tokenize the input
                inputs = self.tokenizer(
                    query,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=getattr(self.strategy.args, 'max_len', 4096)
                ).to(self.credit_model.model.device)
                
                # Get credit predictions
                predicted_credits = self.credit_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                
                # Convert to float32 for better compatibility
                if predicted_credits.dtype == torch.bfloat16:
                    predicted_credits = predicted_credits.float()
                
                # Convert to list and append
                credit_scores.append(predicted_credits.cpu().numpy().tolist())
        
        return credit_scores

    def get_credit_scores_batch(self, queries, batch_size=32):
        """
        批量计算信用分数，提高效率
        """
        if not hasattr(self, 'credit_model'):
            raise RuntimeError("Credit model not initialized")
            
        all_credit_scores = []
        
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_queries,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=getattr(self.strategy.args, 'max_len', 4096)
                ).to(self.credit_model.model.device)
                
                # Get batch predictions
                predicted_credits = self.credit_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                
                # Convert to float32 for better compatibility
                if predicted_credits.dtype == torch.bfloat16:
                    predicted_credits = predicted_credits.float()
                
                # Convert to list
                batch_scores = predicted_credits.cpu().numpy().tolist()
                all_credit_scores.extend(batch_scores)
        
        return all_credit_scores