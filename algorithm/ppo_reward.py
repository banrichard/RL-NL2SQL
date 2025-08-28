"""
PPO阶段的奖励函数
不执行SQL，直接使用reward model输出结合credit model
"""

import torch
from typing import List, Dict, Any, Optional


def ppo_reward_func_without_sql(
    queries: List[str], 
    prompts: List[str], 
    labels: List[str],
    **kwargs
) -> Dict[str, Any]:
    """
    PPO阶段的奖励函数，不执行SQL
    直接使用reward model的输出结合credit model
    
    Args:
        queries: 完整的查询序列 (prompt + generated SQL)
        prompts: 输入prompt序列
        labels: 标签序列 (ground truth SQL)
        **kwargs: 其他参数，包括reward_model_scores
        
    Returns:
        包含最终奖励的字典
    """
    
    # 从kwargs中获取reward model分数
    reward_model_scores = kwargs.get("reward_model_scores", None)
    
    # 如果没有传入reward model分数，使用默认值
    if reward_model_scores is None:
        reward_model_scores = [0.0] * len(queries)
    
    # 确保长度一致
    assert len(queries) == len(reward_model_scores), \
        f"Length mismatch: queries={len(queries)}, scores={len(reward_model_scores)}"
    
    # 获取Credit集成器
    from algorithm.simple_credit_integration import get_credit_integrator
    credit_integrator = get_credit_integrator()
    
    if credit_integrator is not None and kwargs.get("enable_credit_shaping", True):
        # 使用Credit Model增强奖励
        enhanced_data = credit_integrator.enhance_rewards(
            sequences=queries,
            original_rewards=reward_model_scores,
            credit_weight=kwargs.get("credit_weight", 0.1),
            aggregation_method=kwargs.get("credit_aggregation", "mean")
        )
        
        # 添加PPO相关的额外日志
        enhanced_data["extra_logs"].update({
            "reward_source": "reward_model_with_credit",
            "sql_execution": False,
            "base_reward_model_mean": sum(reward_model_scores) / len(reward_model_scores),
            "base_reward_model_std": _compute_std(reward_model_scores),
        })
        
        return enhanced_data
    else:
        # 如果没有Credit Model，直接使用reward model的分数
        return {
            "rewards": reward_model_scores,
            "scores": reward_model_scores,  # 用于dynamic filtering
            "extra_logs": {
                "reward_source": "reward_model_only",
                "sql_execution": False,
                "base_reward_model_mean": sum(reward_model_scores) / len(reward_model_scores),
                "base_reward_model_std": _compute_std(reward_model_scores),
            }
        }


def ppo_reward_func_hybrid(
    queries: List[str], 
    prompts: List[str], 
    labels: List[str],
    use_sql_execution: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    混合模式的奖励函数，可以选择是否执行SQL
    
    Args:
        queries: 完整的查询序列
        prompts: 输入prompt序列  
        labels: 标签序列
        use_sql_execution: 是否执行SQL获得真实分数
        **kwargs: 其他参数，包括reward_model_scores
        
    Returns:
        包含最终奖励的字典
    """
    
    if use_sql_execution:
        # 使用原始的SQL执行方式
        from algorithm.reward import reward_func as original_reward_func
        return original_reward_func(queries, prompts, labels, **kwargs)
    else:
        # 使用PPO模式（不执行SQL）
        return ppo_reward_func_without_sql(
            queries, prompts, labels, **kwargs
        )


def create_ppo_reward_wrapper(
    original_reward_func,
    enable_credit: bool = True,
    credit_weight: float = 0.1,
    credit_aggregation: str = "mean"
):
    """
    创建PPO阶段的奖励函数包装器
    
    Args:
        original_reward_func: 原始的奖励函数
        enable_credit: 是否启用Credit Model
        credit_weight: Credit权重
        credit_aggregation: Credit聚合方法
        
    Returns:
        包装后的奖励函数
    """
    
    def wrapped_reward_func(queries, prompts, labels, **kwargs):
        # 更新kwargs中的credit相关配置
        kwargs.update({
            "enable_credit_shaping": enable_credit,
            "credit_weight": credit_weight,
            "credit_aggregation": credit_aggregation
        })
        
        # 检查是否有预计算的reward model分数
        reward_model_scores = kwargs.pop("reward_model_scores", None)
        
        if reward_model_scores is not None:
            # 如果有预计算的分数，使用PPO模式
            return ppo_reward_func_without_sql(
                queries, prompts, labels, **kwargs
            )
        else:
            # 否则调用原始奖励函数
            return original_reward_func(queries, prompts, labels, **kwargs)
    
    return wrapped_reward_func


def _compute_std(values: List[float]) -> float:
    """计算标准差"""
    if len(values) <= 1:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


# 全局配置
_ppo_reward_config = {
    "use_sql_execution": False,  # PPO阶段默认不执行SQL
    "enable_credit": True,
    "credit_weight": 0.1,
    "credit_aggregation": "mean"
}

def set_ppo_reward_config(**config):
    """设置PPO奖励函数的全局配置"""
    global _ppo_reward_config
    _ppo_reward_config.update(config)

def get_ppo_reward_config():
    """获取PPO奖励函数的全局配置"""
    return _ppo_reward_config.copy()
