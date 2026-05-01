"""
Token-level PRM reward function for PPO training.
Loaded by OpenRLHF via:
    --remote_rm_url /data/banlujie/RL-NL2SQL/algorithm/reward_func_token_level.py

Main idea:
1. Extract SQL from <answer>```sql ... ```</answer>
2. Split SQL into tokens
3. For each token t_i, score RM input:
      PRM_PROMPT(schema, question, prefix_tokens[0..i-1], token=t_i)
4. Return dense per-token scores in extra_logs["token_scores"]
5. Return a scalar aggregated reward by default for PPO compatibility
"""

import math
import os
import re
import sys
import threading
from typing import Any, Dict, List, Tuple

import torch


# ── repo root on sys.path so dataset/ subpackage is importable ────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_REPO_ROOT, os.path.join(_REPO_ROOT, "dataset")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── config via env vars ───────────────────────────────────────────────────────
_RM_PATH = os.environ.get("REWARD_MODEL_PATH", "/home/banlujie/RL-NL2SQL/ckpt/qwen3-0.6b-rm")
_RM_MAX_LEN = int(os.environ.get("REWARD_MODEL_MAX_LEN", "2048"))
_RM_MAX_MEMORY = os.environ.get("REWARD_MODEL_MAX_MEMORY", "").strip()
_RM_GPU_MEMORY_UTIL = os.environ.get("REWARD_MODEL_GPU_MEMORY_UTILIZATION", "").strip()
_NONFINITE_REWARD_FALLBACK = float(os.environ.get("REWARD_NONFINITE_FALLBACK", "-1.0"))
_FORMAT_INVALID_REWARD = float(os.environ.get("REWARD_FORMAT_INVALID_REWARD", "-1.0"))
_REWARD_CLIP_LOW = float(os.environ.get("REWARD_FALLBACK_CLIP_LOW", "-1.0"))
_REWARD_CLIP_HIGH = float(os.environ.get("REWARD_FALLBACK_CLIP_HIGH", "1.0"))
_SIGMOID_TEMPERATURE = float(os.environ.get("REWARD_TOKEN_SIGMOID_TEMPERATURE", "1.0"))
_SIGMOID_BIAS = float(os.environ.get("REWARD_TOKEN_SIGMOID_BIAS", "0.5"))
_TOKEN_AGGREGATION = os.environ.get("REWARD_TOKEN_AGGREGATION", "mean").strip().lower()
# 0: keep rewards scalar; 1: output dense token-level rewards in `rewards` and `scores`.
_RETURN_DENSE = os.environ.get("REWARD_TOKEN_RETURN_DENSE", "0") == "1"


# ── PRM prompt template ───────────────────────────────────────────────────────
_SFT_TMPL = """\
Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
sqlite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Note that while the reasoning process and SQL query need to be enclosed within <think> </think> and <answer> </answer> tags respectively, this should not affect the quality of the SQL generation.
- The answer must contain the SQL query within ```sql ``` tags.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query."""

_PRM_SUFFIX_TOKEN = (
    "\nGenerated SQL Prefix Context:\n{prefix}\n"
    "The next SQL token to be generated is shown below. Please evaluate its logical correctness."
)


# ── lazy singleton: load RM once per process ──────────────────────────────────
_rm_lock = threading.Lock()
_rm_model = None
_rm_tokenizer = None
_rm_device = None


def _resolve_rm_device() -> torch.device:
    pref = os.environ.get("REWARD_MODEL_DEVICE", "auto").strip().lower()
    if pref == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if pref == "auto":
        return torch.device("cuda")
    if pref.isdigit():
        idx = int(pref)
        max_idx = max(torch.cuda.device_count() - 1, 0)
        idx = min(max(idx, 0), max_idx)
        return torch.device(f"cuda:{idx}")
    if pref.startswith("cuda:"):
        try:
            idx = int(pref.split(":", 1)[1])
            max_idx = max(torch.cuda.device_count() - 1, 0)
            idx = min(max(idx, 0), max_idx)
            return torch.device(f"cuda:{idx}")
        except Exception:
            return torch.device("cuda")
    return torch.device("cuda")


def _build_rm_memory_limit(device: torch.device):
    if device.type != "cuda":
        return None

    device_idx = device.index if device.index is not None else 0
    if _RM_MAX_MEMORY:
        return {device_idx: _RM_MAX_MEMORY}

    if not _RM_GPU_MEMORY_UTIL:
        return None

    try:
        fraction = float(_RM_GPU_MEMORY_UTIL)
    except ValueError:
        return None

    # Keep RM memory budgeting independent from vLLM's gpu_memory_utilization.
    fraction = min(max(fraction, 0.01), 0.99)
    total_bytes = torch.cuda.get_device_properties(device_idx).total_memory
    budget_gib = max(int((total_bytes * fraction) // (1024**3)), 1)
    return {device_idx: f"{budget_gib}GiB"}


def _load_rm() -> None:
    global _rm_model, _rm_tokenizer, _rm_device
    with _rm_lock:
        if _rm_model is not None:
            return
        from transformers import AutoTokenizer
        from openrlhf.models import get_llm_for_sequence_regression

        _rm_tokenizer = AutoTokenizer.from_pretrained(_RM_PATH, trust_remote_code=True)
        _rm_device = _resolve_rm_device()
        if _rm_device.type == "cuda":
            try:
                torch.cuda.set_device(_rm_device)
            except Exception:
                pass
        load_kwargs = {}
        max_memory = _build_rm_memory_limit(_rm_device)
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        model = get_llm_for_sequence_regression(
            _RM_PATH,
            "reward",
            normalize_reward=False,
            attn_implementation="flash_attention_2",
            param_dtype="bf16",
            value_head_prefix="score",
            device_map=str(_rm_device),
            **load_kwargs,
        )
        model.eval()
        _rm_model = model


def _score_texts(texts: List[str]) -> List[float]:
    _load_rm()
    enc = _rm_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_RM_MAX_LEN,
    )
    input_ids = enc["input_ids"].to(_rm_device)
    attention_mask = enc["attention_mask"].to(_rm_device)
    with torch.no_grad():
        rewards = _rm_model(input_ids, attention_mask)
    raw = rewards.float().reshape(-1).cpu().tolist()
    return [v if math.isfinite(v) else _NONFINITE_REWARD_FALLBACK for v in raw]


def _signed_sigmoid_score(raw_score: float) -> float:
    if not math.isfinite(raw_score):
        return _NONFINITE_REWARD_FALLBACK
    z = (_SIGMOID_TEMPERATURE * float(raw_score)) + _SIGMOID_BIAS
    z = max(min(z, 40.0), -40.0)
    s = 1.0 / (1.0 + math.exp(-z))
    return (2.0 * s) - 1.0


def _clip_reward(x: float) -> float:
    return max(_REWARD_CLIP_LOW, min(_REWARD_CLIP_HIGH, x))


def _extract_generated_segment(query: str) -> str:
    if "<|im_start|>assistant\n" in query:
        return query.rsplit("<|im_start|>assistant\n", 1)[1]
    if "\nassistant\n" in query:
        return query.rsplit("\nassistant\n", 1)[1]
    if "assistant\n" in query:
        return query.rsplit("assistant\n", 1)[1]
    if query.startswith("assistant\n"):
        return query[len("assistant\n") :]
    return query


def _extract_sql_with_meta(query: str) -> Tuple[str, Dict[str, Any]]:
    generated = _extract_generated_segment(query)
    q_lower = generated.lower()
    has_answer_open = "<answer>" in q_lower
    has_answer_close = "</answer>" in q_lower
    has_answer_tag = has_answer_open and has_answer_close

    answer_blocks = re.findall(r"<answer>(.*?)</answer>", generated, re.DOTALL | re.IGNORECASE)
    for block in reversed(answer_blocks):
        sql_blocks = re.findall(r"```sql\s*(.*?)\s*```", block, re.DOTALL | re.IGNORECASE)
        for sql in reversed(sql_blocks):
            sql = sql.strip()
            if sql:
                return sql, {"has_answer_tag": has_answer_tag}
    return "", {"has_answer_tag": has_answer_tag}


def _parse_chat_prompt(prompt: str):
    sys_m = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    system_content = sys_m.group(1) if sys_m else ""
    schema_m = re.search(
        r"Database Schema:\n(.*?)(?=\n\nExternal Knowledge:|\Z)", system_content, re.DOTALL
    )
    schema_str = schema_m.group(1).strip() if schema_m else ""
    user_m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    question = user_m.group(1).strip() if user_m else ""
    return schema_str, question


def _build_sft_prompt(schema_str: str, question: str) -> str:
    return _SFT_TMPL.format(db_details=schema_str, question=question).strip()


def _build_prm_input_token(sft_prompt: str, prefix_sql: str, token: str) -> str:
    prm = sft_prompt + _PRM_SUFFIX_TOKEN.format(prefix=prefix_sql if prefix_sql else "(none yet)")
    return prm + "\n" + token


def _tokenize_sql(sql: str) -> List[str]:
    sql = (sql or "").strip()
    if not sql:
        return []
    try:
        from sqlglot.tokens import Tokenizer

        tokens = Tokenizer(dialect="sqlite").tokenize(sql)
        texts = [t.text for t in tokens if getattr(t, "text", "")]
        if texts:
            return texts
    except Exception:
        pass
    # fallback tokenizer
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|!=|<=|>=|<>|==|:=|::|[(),.*=<>+\-/%]|'[^']*'|\"[^\"]*\"", sql)


def _tokens_to_prefix(tokens: List[str]) -> str:
    if not tokens:
        return ""
    out: List[str] = []
    for tok in tokens:
        if not out:
            out.append(tok)
            continue
        if tok in {",", ")", ";"}:
            out[-1] = out[-1] + tok
        elif tok in {"(", "."}:
            out.append(tok)
        elif out[-1] in {"(", "."}:
            out[-1] = out[-1] + tok
        else:
            out.append(" " + tok)
    return "".join(out)


def _aggregate(scores: List[float]) -> float:
    if not scores:
        return _FORMAT_INVALID_REWARD
    if _TOKEN_AGGREGATION == "min":
        out = min(scores)
    elif _TOKEN_AGGREGATION == "max":
        out = max(scores)
    elif _TOKEN_AGGREGATION == "last":
        out = scores[-1]
    elif _TOKEN_AGGREGATION == "sum":
        out = sum(scores)
    else:
        out = sum(scores) / len(scores)
    return out if math.isfinite(out) else _NONFINITE_REWARD_FALLBACK


def _first_or_list(xs: List[Any]) -> Any:
    return xs[0] if len(xs) == 1 else xs


def reward_func(queries: List[str], prompts: List[str], labels: List[str], **kwargs) -> Dict[str, Any]:
    del labels, kwargs
    assert len(queries) == len(prompts), "queries/prompts length mismatch"

    all_token_texts: List[List[str]] = []
    all_token_scores_raw: List[List[float]] = []
    all_token_scores: List[List[float]] = []
    all_scalar_rewards: List[float] = []
    all_sqls: List[str] = []
    all_format_valid: List[float] = []

    for query, prompt in zip(queries, prompts):
        sql, format_meta = _extract_sql_with_meta(query)
        all_sqls.append(sql)

        if not sql:
            all_token_texts.append([])
            all_token_scores_raw.append([])
            all_token_scores.append([])
            all_scalar_rewards.append(_FORMAT_INVALID_REWARD)
            all_format_valid.append(0.0 if not format_meta.get("has_answer_tag", False) else 1.0)
            continue

        token_texts = _tokenize_sql(sql)
        if not token_texts:
            all_token_texts.append([])
            all_token_scores_raw.append([])
            all_token_scores.append([])
            all_scalar_rewards.append(_FORMAT_INVALID_REWARD)
            all_format_valid.append(0.0)
            continue

        schema_str, question = _parse_chat_prompt(prompt)
        sft_prompt = _build_sft_prompt(schema_str, question)

        rm_inputs: List[str] = []
        for i, tok in enumerate(token_texts):
            prefix_sql = _tokens_to_prefix(token_texts[:i])
            rm_inputs.append(_build_prm_input_token(sft_prompt, prefix_sql, tok))

        token_scores_raw = _score_texts(rm_inputs)
        token_scores = [_signed_sigmoid_score(s) for s in token_scores_raw]
        scalar_reward = _clip_reward(_aggregate(token_scores))

        all_token_texts.append(token_texts)
        all_token_scores_raw.append([float(s) for s in token_scores_raw])
        all_token_scores.append([float(s) for s in token_scores])
        all_scalar_rewards.append(float(scalar_reward))
        all_format_valid.append(1.0)

    format_valid_rate = sum(all_format_valid) / max(len(all_format_valid), 1)
    token_count_mean = sum(len(x) for x in all_token_texts) / max(len(all_token_texts), 1)
    scalar_reward_mean = sum(all_scalar_rewards) / max(len(all_scalar_rewards), 1)

    if _RETURN_DENSE:
        rewards_out = _first_or_list(all_token_scores)
        scores_out = rewards_out
    else:
        rewards_out = _first_or_list(all_scalar_rewards)
        scores_out = rewards_out

    return {
        "rewards": rewards_out,
        "scores": scores_out,
        "extra_logs": {
            "reward_source": "token_level_prm",
            "token_aggregation": _TOKEN_AGGREGATION,
            "return_dense_rewards": 1.0 if _RETURN_DENSE else 0.0,
            "format_valid_rate": format_valid_rate,
            "token_count_mean": token_count_mean,
            "scalar_reward_mean": scalar_reward_mean,
            "extracted_sql": _first_or_list(all_sqls),
            "token_texts": _first_or_list(all_token_texts),
            "token_scores_raw": _first_or_list(all_token_scores_raw),
            "token_scores": _first_or_list(all_token_scores),
        },
    }
