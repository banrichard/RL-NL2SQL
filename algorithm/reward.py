import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import Counter
from pathlib import Path

from database.sql_executor import cached_execute_sql_with_timeout, SQLExecutionResultType

FORMAT_REWARD = 1
EXEC_REWARD = 2
RESULT_REWARD = 3

LIMIT_LENGTH = 2048


def _extract_response_from_query(query: str) -> str:
    assistant_index = query.find("assistant\n")
    if assistant_index != -1:
        response_start_index = assistant_index + len("assistant\n")
        return query[response_start_index:].strip()
    return query.strip()


def _extract_sql(text: str) -> str:
    if not text:
        return ""

    # 优先提取 ```sql ... ``` 代码块
    sql_blocks = re.findall(r"```sql\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if sql_blocks:
        return sql_blocks[-1].strip().rstrip(";")

    # 回退到任意代码块
    code_blocks = re.findall(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip().rstrip(";")

    # 再回退到 SELECT/WITH 起始的 SQL 片段
    sql_match = re.search(r"((?:WITH|SELECT)\b[\s\S]*)", text, flags=re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip().rstrip(";")

    return text.strip().rstrip(";")


def _extract_gold_sql(label: Any) -> str:
    if isinstance(label, str):
        return _extract_sql(label)
    if not isinstance(label, dict):
        return ""

    for key in ("sql", "SQL", "gold_sql", "label", "answer"):
        if key in label and label[key]:
            return _extract_sql(str(label[key]))
    return ""


def _resolve_db_path(label: Any, db_path: Optional[Union[str, Path]] = None, database_root_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    candidate_paths: List[Path] = []

    if db_path:
        candidate_paths.append(Path(db_path))

    if isinstance(label, dict):
        for key in ("db_path", "database_path", "db_file", "sqlite_path"):
            value = label.get(key)
            if value:
                candidate_paths.append(Path(value))

        db_id = label.get("db_id") or label.get("database_id")
        if db_id and database_root_dir:
            root = Path(database_root_dir)
            candidate_paths.append(root / str(db_id) / f"{db_id}.sqlite")
            candidate_paths.append(root / f"{db_id}.sqlite")

    for path in candidate_paths:
        if path.exists():
            return path
    return None


def _normalize_rows(rows: Optional[Sequence[Tuple[Any, ...]]]) -> Counter:
    if rows is None:
        return Counter()

    normalized = []
    for row in rows:
        normalized.append(tuple(str(v) for v in row))
    return Counter(normalized)


def reward_func(queries, prompts, labels, **kwargs):
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

    rewards: List[float] = []
    execution_success = 0

    db_paths = kwargs.get("db_paths")
    default_db_path = kwargs.get("db_path") or kwargs.get("database_path")
    database_root_dir = kwargs.get("database_root_dir") or kwargs.get("db_root")

    for idx, query in enumerate(queries):
        response = _extract_response_from_query(query)
        label = labels[idx] if idx < len(labels) else ""

        item_db_path = None
        if isinstance(db_paths, (list, tuple)) and idx < len(db_paths):
            item_db_path = db_paths[idx]

        reward = execution_reward(
            answer=response,
            label=label,
            query=query,
            db_path=item_db_path or default_db_path,
            database_root_dir=database_root_dir,
        )
        rewards.append(reward)
        if reward > 0.0:
            execution_success += 1

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

    return {
        "rewards": rewards,
        "scores": rewards,
        "extra_logs": {
            "reward_source": "sql_execution_exact_match",
            "sql_execution": True,
            "execution_success_rate": execution_success / len(rewards) if rewards else 0.0,
            "mean_execution_reward": mean_reward,
        },
    }



def execution_reward(
    answer: str,
    label: Any,
    query: str,
    db_path: Optional[Union[str, Path]],
    database_root_dir: Optional[Union[str, Path]] = None,
) -> float:
    pred_sql = _extract_sql(answer)
    if not pred_sql:
        pred_sql = _extract_sql(query)

    gold_sql = _extract_gold_sql(label)
    resolved_db_path = _resolve_db_path(label, db_path=db_path, database_root_dir=database_root_dir)

    if not pred_sql or not gold_sql or resolved_db_path is None:
        return 0.0

    pred_result = cached_execute_sql_with_timeout(str(resolved_db_path), pred_sql)
    gold_result = cached_execute_sql_with_timeout(str(resolved_db_path), gold_sql)

    if pred_result.result_type != SQLExecutionResultType.SUCCESS:
        return 0.0
    if gold_result.result_type != SQLExecutionResultType.SUCCESS:
        return 0.0

    return 1.0 if _normalize_rows(pred_result.result) == _normalize_rows(gold_result.result) else 0.0
