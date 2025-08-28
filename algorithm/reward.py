try:
    from database.sql_executor import cached_execute_sql_with_timeout, SQLExecutionResultType, \
        is_valid_execution_result
    SQL_EXECUTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SQL executor not available: {e}")
    SQL_EXECUTOR_AVAILABLE = False
    
from typing import Dict, Any, List
from collections import defaultdict
from pathlib import Path

# Import Credit Model integration
try:
    from algorithm.simple_credit_integration import get_credit_integrator, init_credit_integrator
    CREDIT_INTEGRATION_AVAILABLE = True
except ImportError as e:
    CREDIT_INTEGRATION_AVAILABLE = False
    print(f"Credit Model integration not available: {e}")

FORMAT_REWARD = 1
EXEC_REWARD = 2
RESULT_REWARD = 3

LIMIT_LENGTH = 2048


def reward_func(queries, prompts, labels, **kwargs):
    """
    Reward function for calculating rewards of model outputs.
    Enhanced with Credit Model integration for fine-grained credit assignment.

    Args:
        queries (torch.Tensor): Complete text sequences containing prompts and responses
        prompts (torch.Tensor): Input prompt sequences
        labels (torch.Tensor): Ground truth answer sequences
        **kwargs: Additional optional parameters including:
            - credit_model_path: Path to trained Credit Model
            - credit_weight: Weight for credit-based reward adjustment
            - credit_aggregation: Method to aggregate credit scores

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - rewards: Reward values used for calculating advantage function
            - scores: Reward values in range [0,1] used for dynamic filtering
            - extra_logs: Additional information to be logged in wandb
    """

    responses = []

    # Extract responses from queries
    for query in queries:
        assistant_index = query.find("assistant\n")
        if assistant_index != -1:
            # 计算 "assistant\n" 后面内容的起始位置
            response_start_index = assistant_index + len("assistant\n")
            # 提取从 response_start_index 开始到结尾的字符串，并去除前后空白字符
            response = query[response_start_index:].strip()
            # 将提取出的 response 添加到 responses 列表中
            responses.append(response)
        else:
            # 如果没有找到 "assistant\n"，可以选择抛出错误、警告或者跳过
            # 这里我们选择跳过该 query 或者可以给一个默认值
            responses.append("")  # 或者使用 pass 跳过

    # Calculate basic rewards (you can implement your SQL execution logic here)
    # For now, using a simple placeholder
    import torch
    basic_rewards = torch.randint(0, 2, (len(queries),)).float().tolist()
    
    # TODO: Implement your actual reward logic here
    # For example:
    # basic_rewards = []
    # for response, label in zip(responses, labels):
    #     reward = calculate_sql_execution_reward(response, label)
    #     basic_rewards.append(reward)
    
    # Basic reward data
    reward_data = {
        "rewards": basic_rewards,
        "scores": basic_rewards,  # Use same values for scoring
        "extra_logs": {"basic_reward_mean": sum(basic_rewards) / len(basic_rewards) if basic_rewards else 0.0}
    }
    
    # Credit Model integration
    credit_model_path = kwargs.get("credit_model_path")
    if CREDIT_INTEGRATION_AVAILABLE and credit_model_path:
        try:
            # Initialize Credit integrator if needed
            credit_integrator = get_credit_integrator()
            if credit_integrator is None:
                init_credit_integrator(credit_model_path)
                credit_integrator = get_credit_integrator()
            
            if credit_integrator is not None:
                # Enhance rewards with Credit Model
                enhanced_data = credit_integrator.enhance_rewards(
                    sequences=queries,
                    original_rewards=basic_rewards,
                    credit_weight=kwargs.get("credit_weight", 0.1),
                    aggregation_method=kwargs.get("credit_aggregation", "mean")
                )
                
                # Use enhanced data
                reward_data = enhanced_data
                print(f"Rewards enhanced with Credit Model. Mean credit reward: {enhanced_data['extra_logs']['mean_credit_reward']:.4f}")
            
        except Exception as e:
            print(f"Warning: Failed to apply Credit Model enhancement: {e}")
            # Fall back to basic rewards
    
    return reward_data



def execution_reward(answer,label, query, db_path: Path) -> float:
    pass
