"""
简化版的Credit Model集成
直接在reward计算中加入Credit Model的评估
"""

import torch
from typing import List, Dict, Any
from algorithm.credit_model import CreditModel
from transformers import AutoTokenizer


class SimpleCreditIntegrator:
    """
    简单的Credit Model集成器
    可以直接在现有的reward函数中使用
    """
    
    def __init__(self, credit_model_path: str, device: str = "cuda"):
        """
        初始化Credit Model集成器
        
        Args:
            credit_model_path: 训练好的Credit Model路径
            device: 运行设备
        """
        self.device = device
        
        # 加载Credit Model
        print(f"Loading Credit Model from {credit_model_path}")
        self.credit_model = CreditModel.from_pretrained(credit_model_path)
        self.credit_model = self.credit_model.to(device)
        self.credit_model.eval()
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(credit_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Credit Model loaded successfully!")
    
    def compute_credit_scores(self, sequences: List[str], max_length: int = 4096) -> List[List[float]]:
        """
        计算序列的信用分数
        
        Args:
            sequences: 完整的序列列表 (prompt + response)
            max_length: 最大序列长度
            
        Returns:
            每个序列的信用分数向量列表
        """
        credit_scores = []
        
        with torch.no_grad():
            for sequence in sequences:
                try:
                    # Tokenize
                    inputs = self.tokenizer(
                        sequence,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(self.device)
                    
                    # Get credit predictions
                    predicted_credits = self.credit_model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
                    
                    # Convert to float32 and then to list for better compatibility
                    if predicted_credits.dtype == torch.bfloat16:
                        predicted_credits = predicted_credits.float()
                    
                    scores = predicted_credits.cpu().squeeze().tolist()
                    if isinstance(scores, float):
                        scores = [scores]  # 确保是列表格式
                    
                    credit_scores.append(scores)
                    
                except Exception as e:
                    print(f"Warning: Failed to compute credit for sequence: {e}")
                    credit_scores.append([0.0] * 8)  # 默认8维零向量
        
        return credit_scores
    
    def aggregate_credit_reward(self, credit_scores: List[float], method: str = "mean") -> float:
        """
        将信用分数聚合为单一的奖励值
        
        Args:
            credit_scores: 信用分数向量
            method: 聚合方法 ("mean", "max", "sum", "weighted_sum")
            
        Returns:
            聚合后的信用奖励
        """
        if not credit_scores:
            return 0.0
        
        if method == "mean":
            return sum(credit_scores) / len(credit_scores)
        elif method == "max":
            return max(credit_scores)
        elif method == "sum":
            return sum(credit_scores)
        elif method == "weighted_sum":
            # 给不同位置的信用分数不同权重
            weights = [1.0 / (i + 1) for i in range(len(credit_scores))]
            return sum(s * w for s, w in zip(credit_scores, weights))
        else:
            return sum(credit_scores) / len(credit_scores)  # 默认使用mean
    
    def enhance_rewards(self, 
                       sequences: List[str], 
                       original_rewards: List[float],
                       credit_weight: float = 0.1,
                       aggregation_method: str = "mean") -> Dict[str, Any]:
        """
        使用Credit Model增强原始奖励
        
        Args:
            sequences: 完整序列列表
            original_rewards: 原始奖励列表
            credit_weight: Credit奖励的权重
            aggregation_method: 信用分数聚合方法
            
        Returns:
            增强后的奖励数据
        """
        # 计算信用分数
        credit_scores_batch = self.compute_credit_scores(sequences)
        
        # 聚合信用分数为奖励
        credit_rewards = [
            self.aggregate_credit_reward(scores, aggregation_method) 
            for scores in credit_scores_batch
        ]
        
        # 增强原始奖励
        enhanced_rewards = [
            original + credit_weight * credit 
            for original, credit in zip(original_rewards, credit_rewards)
        ]
        
        return {
            "rewards": enhanced_rewards,
            "credit_rewards": credit_rewards,
            "original_rewards": original_rewards,
            "scores": enhanced_rewards,  # 用于dynamic filtering
            "extra_logs": {
                "mean_credit_reward": sum(credit_rewards) / len(credit_rewards) if credit_rewards else 0.0,
                "max_credit_reward": max(credit_rewards) if credit_rewards else 0.0,
                "min_credit_reward": min(credit_rewards) if credit_rewards else 0.0,
                "credit_weight": credit_weight,
                "credit_aggregation": aggregation_method,
            }
        }


# 全局Credit集成器实例
_global_credit_integrator = None

def init_credit_integrator(credit_model_path: str, device: str = "cuda"):
    """
    初始化全局Credit集成器
    在reward函数开始时调用一次
    """
    global _global_credit_integrator
    if _global_credit_integrator is None and credit_model_path:
        _global_credit_integrator = SimpleCreditIntegrator(credit_model_path, device)
    return _global_credit_integrator

def get_credit_integrator():
    """
    获取全局Credit集成器
    """
    return _global_credit_integrator

def credit_enhanced_reward_func(queries, prompts, labels, **kwargs):
    """
    集成Credit Model的奖励函数示例
    可以直接替换algorithm/reward.py中的reward_func
    
    Args:
        queries: 完整的查询序列
        prompts: 输入prompt序列
        labels: 标签序列
        **kwargs: 其他参数，包括credit相关配置
        
    Returns:
        包含Credit增强奖励的字典
    """
    # 这里可以调用你原来的reward_func获取基础奖励
    from algorithm.reward import reward_func as original_reward_func
    
    # 获取原始奖励
    original_reward_data = original_reward_func(queries, prompts, labels)
    original_rewards = original_reward_data.get("rewards", [0.0] * len(queries))
    
    # 获取Credit集成器
    credit_integrator = get_credit_integrator()
    
    if credit_integrator is not None:
        # 使用Credit Model增强奖励
        enhanced_data = credit_integrator.enhance_rewards(
            sequences=queries,
            original_rewards=original_rewards,
            credit_weight=kwargs.get("credit_weight", 0.1),
            aggregation_method=kwargs.get("credit_aggregation", "mean")
        )
        
        # 合并额外的日志信息
        if "extra_logs" in original_reward_data:
            enhanced_data["extra_logs"].update(original_reward_data["extra_logs"])
        
        return enhanced_data
    else:
        # 如果没有Credit Model，返回原始奖励
        return original_reward_data


def create_credit_enhanced_ppo_trainer(
    base_trainer_class,
    credit_model_actor_group=None,
    credit_weight: float = 0.1,
    enable_credit_shaping: bool = True,
    credit_aggregation: str = "weighted_sum"
):
    """
    简化版本：不使用继承，直接返回原始trainer类
    避免Ray actor继承问题
    
    同时配置PPO阶段的奖励函数，使其不执行SQL
    """
    if not credit_model_actor_group or not enable_credit_shaping:
        print("Credit shaping disabled or no credit model provided")
        return base_trainer_class
    
    print(f"Simple credit integration enabled: weight={credit_weight}, aggregation={credit_aggregation}")
    
    # 在trainer类上设置credit配置，供后续使用
    base_trainer_class._credit_config = {
        'credit_model_actor_group': credit_model_actor_group,
        'credit_weight': credit_weight,
        'credit_aggregation': credit_aggregation,
        'enable_credit_shaping': enable_credit_shaping
    }
    
    # 配置PPO阶段的奖励函数
    from algorithm.ppo_reward import set_ppo_reward_config
    set_ppo_reward_config(
        use_sql_execution=False,  # PPO阶段不执行SQL
        enable_credit=enable_credit_shaping,
        credit_weight=credit_weight,
        credit_aggregation=credit_aggregation
    )
    
    print("PPO reward function configured to use reward model + credit model (no SQL execution)")
    
    return base_trainer_class


def setup_ppo_reward_function(credit_model_path: str = None, **kwargs):
    """
    设置PPO阶段的奖励函数
    
    Args:
        credit_model_path: Credit Model路径
        **kwargs: 其他配置参数
    """
    # 初始化Credit集成器（如果提供了路径）
    if credit_model_path:
        init_credit_integrator(credit_model_path, kwargs.get("device", "cuda"))
    
    # 配置PPO奖励函数
    from algorithm.ppo_reward import set_ppo_reward_config
    set_ppo_reward_config(**kwargs)
    
    print(f"PPO reward function setup completed with config: {kwargs}")
