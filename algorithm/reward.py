from database.sql_executor import cached_execute_sql_with_timeout, SQLExecutionResultType, \
    is_valid_execution_result
from typing import Dict, Any, List
from collections import defaultdict
from pathlib import Path

FORMAT_REWARD = 1
EXEC_REWARD = 2
RESULT_REWARD = 3

LIMIT_LENGTH = 2048


def reward_func(queries, prompts, labels):
    """
    Reward function for calculating rewards of model outputs.

    Args:
        queries (torch.Tensor): Complete text sequences containing prompts and responses
        prompts (torch.Tensor): Input prompt sequences
        labels (torch.Tensor): Ground truth answer sequences
        **kwargs: Additional optional parameters

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - rewards: Reward values used for calculating advantage function
            - scores: Reward values in range [0,1] used for dynamic filtering
            - extra_logs: Additional information to be logged in wandb
    """

    responses = []

    # reward = torch.randint(0, 2, (len(queries),)).float()
    #
    # return {
    #     "rewards": reward,  # Rewards for advantage calculation
    #     "scores": reward,  # Scores for dynamic filtering (0-1 reward)
    #     "extra_logs": {"dummy_scores": reward},  # Additional logging info for wandb
    # }

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



def execution_reward(answer,label, query, db_path: Path) -> float:
    pass
