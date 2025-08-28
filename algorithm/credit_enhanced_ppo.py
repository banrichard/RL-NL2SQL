"""
Enhanced PPO Trainer with Credit Assignment
基于OpenRLHF的PPOTrainer，集成Credit Model进行细粒度的信用分配
"""

import ray
import torch
from typing import List, Dict, Any, Optional
from openrlhf.trainer.ppo_trainer import PPOTrainer
from algorithm.credit_model_actor import CreditModelActor


class CreditEnhancedPPOTrainer:
    """
    集成Credit Model的PPO训练器
    在原有的PPO训练基础上，添加基于Credit Model的细粒度奖励分配
    """
    
    def __init__(
        self,
        base_ppo_trainer,
        credit_model_actor_group,
        credit_weight: float = 0.1,  # Credit Model奖励的权重
        enable_credit_shaping: bool = True,  # 是否启用信用塑形
        credit_aggregation: str = "weighted_sum",  # 信用聚合方式: "weighted_sum", "max", "mean"
    ):
        self.base_ppo_trainer = base_ppo_trainer
        self.credit_model_actor_group = credit_model_actor_group
        self.credit_weight = credit_weight
        self.enable_credit_shaping = enable_credit_shaping
        self.credit_aggregation = credit_aggregation
        
        print(f"CreditEnhancedPPOTrainer initialized with:")
        print(f"  - Credit weight: {credit_weight}")
        print(f"  - Credit shaping: {enable_credit_shaping}")
        print(f"  - Credit aggregation: {credit_aggregation}")

    def compute_credit_rewards(self, queries: List[str], responses: List[str]) -> List[float]:
        """
        使用Credit Model计算细粒度的信用奖励
        
        Args:
            queries: 完整的查询序列 (prompt + response)
            responses: 响应部分
            
        Returns:
            List of credit-based reward adjustments
        """
        if not self.enable_credit_shaping:
            return [0.0] * len(queries)
        
        try:
            # 从credit model actor group获取信用分数
            # 假设我们只使用第一个actor，也可以扩展为ensemble
            credit_scores_batch = ray.get(
                self.credit_model_actor_group.actors[0].get_credit_scores_batch.remote(queries)
            )
            
            credit_rewards = []
            for i, (credit_scores, response) in enumerate(zip(credit_scores_batch, responses)):
                # 这里可以根据具体需求实现不同的聚合策略
                if self.credit_aggregation == "weighted_sum":
                    # 加权求和：根据响应长度分配权重
                    response_length = len(response.split())
                    if response_length > 0:
                        # 对每个token位置的信用分数进行加权
                        weights = [1.0 / response_length] * min(len(credit_scores), response_length)
                        credit_reward = sum(score * weight for score, weight in zip(credit_scores, weights))
                    else:
                        credit_reward = 0.0
                        
                elif self.credit_aggregation == "max":
                    # 取最大信用分数
                    credit_reward = max(credit_scores) if credit_scores else 0.0
                    
                elif self.credit_aggregation == "mean":
                    # 取平均信用分数
                    credit_reward = sum(credit_scores) / len(credit_scores) if credit_scores else 0.0
                    
                else:
                    raise ValueError(f"Unsupported credit aggregation method: {self.credit_aggregation}")
                
                credit_rewards.append(credit_reward)
            
            return credit_rewards
            
        except Exception as e:
            print(f"Warning: Failed to compute credit rewards: {e}")
            return [0.0] * len(queries)

    def enhance_rewards_with_credit(self, reward_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用Credit Model增强原始奖励
        
        Args:
            reward_data: 原始的奖励数据，包含rewards, scores等
            
        Returns:
            增强后的奖励数据
        """
        if not hasattr(self, '_last_queries') or not hasattr(self, '_last_responses'):
            print("Warning: No query/response data available for credit enhancement")
            return reward_data
        
        # 计算credit rewards
        credit_rewards = self.compute_credit_rewards(self._last_queries, self._last_responses)
        
        # 增强原始奖励
        enhanced_rewards = []
        original_rewards = reward_data.get('rewards', [])
        
        for i, (original_reward, credit_reward) in enumerate(zip(original_rewards, credit_rewards)):
            enhanced_reward = original_reward + self.credit_weight * credit_reward
            enhanced_rewards.append(enhanced_reward)
        
        # 更新奖励数据
        enhanced_reward_data = reward_data.copy()
        enhanced_reward_data['rewards'] = enhanced_rewards
        enhanced_reward_data['credit_rewards'] = credit_rewards  # 记录信用奖励以便分析
        
        # 记录统计信息
        if 'extra_logs' not in enhanced_reward_data:
            enhanced_reward_data['extra_logs'] = {}
        
        enhanced_reward_data['extra_logs'].update({
            'mean_credit_reward': sum(credit_rewards) / len(credit_rewards) if credit_rewards else 0.0,
            'max_credit_reward': max(credit_rewards) if credit_rewards else 0.0,
            'min_credit_reward': min(credit_rewards) if credit_rewards else 0.0,
            'credit_weight': self.credit_weight,
        })
        
        return enhanced_reward_data

    def cache_generation_data(self, queries: List[str], responses: List[str]):
        """
        缓存生成数据，供credit reward计算使用
        """
        self._last_queries = queries.copy()
        self._last_responses = responses.copy()

    def set_credit_weight(self, new_weight: float):
        """
        动态调整credit权重
        """
        self.credit_weight = new_weight
        print(f"Credit weight updated to: {new_weight}")

    def get_training_stats(self) -> Dict[str, Any]:
        """
        获取训练统计信息，包括credit相关的指标
        """
        stats = {
            'credit_weight': self.credit_weight,
            'credit_shaping_enabled': self.enable_credit_shaping,
            'credit_aggregation': self.credit_aggregation,
        }
        
        if hasattr(self, '_last_queries'):
            stats['last_batch_size'] = len(self._last_queries)
        
        return stats


# 创建一个简单的工厂函数，避免Ray actor继承问题
def create_credit_enhanced_ppo_trainer(
    base_trainer_class,
    credit_model_actor_group,
    credit_weight: float = 0.1,
    enable_credit_shaping: bool = True,
    credit_aggregation: str = "weighted_sum"
):
    """
    创建集成Credit Model的PPO训练器
    
    由于Ray不支持actor继承，我们返回原始trainer类，并通过monkey patching增强功能
    """
    
    if credit_model_actor_group is None:
        print("No credit model provided, using standard PPO trainer")
        return base_trainer_class
    
    print("Creating credit-enhanced PPO trainer using method patching...")
    
    # 保存原始方法
    original_init = base_trainer_class.__init__
    
    def enhanced_init(self, *args, **kwargs):
        # 调用原始初始化
        original_init(self, *args, **kwargs)
        
        # 添加credit增强组件
        self.credit_enhancer = CreditEnhancedPPOTrainer(
            base_ppo_trainer=self,
            credit_model_actor_group=credit_model_actor_group,
            credit_weight=credit_weight,
            enable_credit_shaping=enable_credit_shaping,
            credit_aggregation=credit_aggregation
        )
        print("Credit enhancer initialized successfully")
    
    # 修改类的__init__方法
    base_trainer_class.__init__ = enhanced_init
    
    return base_trainer_class
