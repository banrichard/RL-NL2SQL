"""
Execution-only reward function for ablation study.

Loaded by OpenRLHF via:
  --remote_rm_url /data/banlujie/RL-NL2SQL/algorithm/reward_func_exec_only_ablation.py

Reward rule:
  - pred SQL cannot be parsed by sqlglot: -1.0
  - pred SQL parses and execution result equals ground-truth result: +1.0
  - pred SQL parses but execution fails: 0.0
  - pred SQL parses and executes but result mismatches ground truth: +0.1
"""

import json
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import sqlglot

from database.sql_executor import cached_execute_sql_with_timeout, SQLExecutionResultType

_PARSE_FAIL_REWARD = -1.0
_EXEC_ERROR_REWARD = 0.0
_EXEC_MATCH_REWARD = 1.0
_EXEC_MISMATCH_REWARD = 0.1

_BIRD_TABLES_FILES = [
    p.strip()
    for p in os.environ.get(
        "REWARD_BIRD_TABLES_FILES",
        "/data/banlujie/data/bird_dev/dev_tables.json,/data/banlujie/data/bird_train/train/train_tables.json",
    ).split(",")
    if p.strip()
]
_BIRD_DB_ROOTS = [
    p.strip()
    for p in os.environ.get(
        "REWARD_BIRD_DB_ROOTS",
        "/data/banlujie/data/bird_dev/dev_databases,/data/banlujie/data/bird_train/train/train_databases",
    ).split(",")
    if p.strip()
]

_schema_to_db_path: Dict[str, str] = {}
_schema_index_lock = threading.Lock()


def _extract_response_from_query(query: str) -> str:
    if "<|im_start|>assistant\n" in query:
        return query.rsplit("<|im_start|>assistant\n", 1)[1].strip()
    if "\nassistant\n" in query:
        return query.rsplit("\nassistant\n", 1)[1].strip()
    if "assistant\n" in query:
        return query.rsplit("assistant\n", 1)[1].strip()
    return query.strip()


def _extract_sql(text: str) -> str:
    if not text:
        return ""

    sql_blocks = re.findall(r"```sql\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if sql_blocks:
        return sql_blocks[-1].strip().rstrip(";")

    code_blocks = re.findall(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip().rstrip(";")

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


def _parse_chat_prompt(prompt: str) -> Tuple[str, str, str]:
    sys_m = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    system_content = sys_m.group(1) if sys_m else ""

    schema_m = re.search(
        r"Database Schema:\n(.*?)(?=\n\nExternal Knowledge:|\Z)", system_content, re.DOTALL
    )
    schema_str = schema_m.group(1).strip() if schema_m else ""

    ev_m = re.search(r"External Knowledge:\n(.*?)(?=\Z)", system_content, re.DOTALL)
    evidence = ev_m.group(1).strip() if ev_m else ""

    user_m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    question = user_m.group(1).strip() if user_m else ""

    return schema_str, question, evidence


def _normalize_schema_text(schema_str: str) -> str:
    return re.sub(r"\s+", " ", (schema_str or "").strip()).lower()


def _build_schema_db_index() -> Dict[str, str]:
    from dataset.utils import serialize_schema

    index: Dict[str, str] = {}
    for tables_file in _BIRD_TABLES_FILES:
        if not os.path.exists(tables_file):
            continue
        try:
            with open(tables_file, "r", encoding="utf-8") as f:
                tables = json.load(f)
        except Exception:
            continue

        for item in tables:
            db_id = item.get("db_id")
            if not db_id:
                continue
            try:
                schema_text = serialize_schema(item)
            except Exception:
                continue
            key = _normalize_schema_text(schema_text)
            if not key:
                continue
            for root in _BIRD_DB_ROOTS:
                cand = os.path.join(root, db_id, f"{db_id}.sqlite")
                if os.path.exists(cand):
                    index[key] = cand
                    break
    return index


def _resolve_db_path_from_schema(schema_str: str) -> Optional[Path]:
    key = _normalize_schema_text(schema_str)
    if not key:
        return None
    with _schema_index_lock:
        if not _schema_to_db_path:
            _schema_to_db_path.update(_build_schema_db_index())
        resolved = _schema_to_db_path.get(key)
        return Path(resolved) if resolved else None


def _resolve_db_path(
    label: Any,
    prompt: str,
    db_path: Optional[Union[str, Path]] = None,
    database_root_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
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

    schema_str, _, _ = _parse_chat_prompt(prompt)
    return _resolve_db_path_from_schema(schema_str)


def _normalize_rows(rows: Optional[Sequence[Tuple[Any, ...]]]) -> Counter:
    if rows is None:
        return Counter()
    return Counter(tuple(str(v) for v in row) for row in rows)


def _is_sqlglot_parseable(sql: str) -> bool:
    if not sql:
        return False
    try:
        sqlglot.parse_one(sql, read="sqlite")
        return True
    except Exception:
        return False


def _single_reward(
    query: str,
    label: Any,
    prompt: str,
    db_path: Optional[Union[str, Path]],
    database_root_dir: Optional[Union[str, Path]],
) -> Tuple[float, str]:
    answer = _extract_response_from_query(query)
    pred_sql = _extract_sql(answer) or _extract_sql(query)

    if not _is_sqlglot_parseable(pred_sql):
        return _PARSE_FAIL_REWARD, "parse_fail"

    gold_sql = _extract_gold_sql(label)
    resolved_db_path = _resolve_db_path(
        label=label,
        prompt=prompt,
        db_path=db_path,
        database_root_dir=database_root_dir,
    )
    if not gold_sql or resolved_db_path is None:
        return _EXEC_MISMATCH_REWARD, "missing_gt_or_db"

    pred_result = cached_execute_sql_with_timeout(str(resolved_db_path), pred_sql)
    if pred_result.result_type != SQLExecutionResultType.SUCCESS:
        return _EXEC_ERROR_REWARD, "pred_exec_error"

    gold_result = cached_execute_sql_with_timeout(str(resolved_db_path), gold_sql)
    if gold_result.result_type != SQLExecutionResultType.SUCCESS:
        return _EXEC_MISMATCH_REWARD, "gt_exec_error"

    if _normalize_rows(pred_result.result) == _normalize_rows(gold_result.result):
        return _EXEC_MATCH_REWARD, "exec_match"
    return _EXEC_MISMATCH_REWARD, "exec_mismatch"


def reward_func(queries, prompts, labels, **kwargs):
    rewards: List[float] = []
    statuses: List[str] = []

    db_paths = kwargs.get("db_paths")
    default_db_path = kwargs.get("db_path") or kwargs.get("database_path")
    database_root_dir = kwargs.get("database_root_dir") or kwargs.get("db_root")

    for idx, query in enumerate(queries):
        label = labels[idx] if idx < len(labels) else ""
        prompt = prompts[idx] if idx < len(prompts) else ""

        item_db_path = None
        if isinstance(db_paths, (list, tuple)) and idx < len(db_paths):
            item_db_path = db_paths[idx]

        reward, status = _single_reward(
            query=query,
            label=label,
            prompt=prompt,
            db_path=item_db_path or default_db_path,
            database_root_dir=database_root_dir,
        )
        rewards.append(reward)
        statuses.append(status)

    n = len(rewards) if rewards else 1
    parse_fail_count = sum(1 for s in statuses if s == "parse_fail")
    exec_match_count = sum(1 for s in statuses if s == "exec_match")
    pred_exec_error_count = sum(1 for s in statuses if s == "pred_exec_error")

    return {
        "rewards": rewards,
        "scores": rewards,
        "extra_logs": {
            "reward_source": "exec_only_ablation",
            "parse_fail_reward": _PARSE_FAIL_REWARD,
            "exec_error_reward": _EXEC_ERROR_REWARD,
            "exec_match_reward": _EXEC_MATCH_REWARD,
            "parse_fail_rate": parse_fail_count / n,
            "exec_match_rate": exec_match_count / n,
            "pred_exec_error_rate": pred_exec_error_count / n,
        },
    }
