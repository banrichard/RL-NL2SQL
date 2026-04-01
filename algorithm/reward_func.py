"""
PRM-based reward function for PPO training.
Loaded by OpenRLHF via --remote_rm_url /path/to/reward_func.py

Interface (called by SingleTurnAgentExecutor._fetch_rewards_via_func):
    reward_func(queries, prompts, labels) -> {"rewards": ..., "scores": ..., "extra_logs": {}}

- queries : List[str]  full decoded text (actor chat template + <think>...</think> + <answer>SQL</answer>)
- prompts : List[str]  original PPO prompt (system message contains schema; user message contains question)
- labels  : List[str]  ground truth SQL (from --label_key sql, currently unused)

Strategy (Plan B: true PRM clause-level scoring):
  1. Extract the generated SQL from <answer>```sql ... ```</answer> in query
  2. Parse the SQL with sqlglot, decompose into clauses per CLAUSE_ORDER
  3. Parse schema + question out of the chat-template prompt
  4. For each clause c_i:
       prefix = concat of clause texts [0..i-1]
       prm_prompt = build_prm_prompt(sft_prompt, prefix)  -- matches PRM training format
       rm_input = prm_prompt + "\\n" + c_i
       score_i = RM(rm_input)
  5. Aggregate: reward = mean(scores)  (or min, tunable via env var)
"""

import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional

import torch

# ── repo root on sys.path so dataset/ subpackage is importable ────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_REPO_ROOT, os.path.join(_REPO_ROOT, "dataset")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── config via env vars ───────────────────────────────────────────────────────
_RM_PATH      = os.environ.get("REWARD_MODEL_PATH",
                               "/home/banlujie/RL-NL2SQL/ckpt/qwen3-0.6b-rm")
_RM_MAX_LEN   = int(os.environ.get("REWARD_MODEL_MAX_LEN", "2048"))
# Aggregation: "mean" | "min" | "last"
_AGGREGATION  = os.environ.get("REWARD_FUNC_AGGREGATION", "mean")

# ── PRM prompt template (matches build_v1_reward_dataset.py:build_prm_prompt) ─
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

# PRM prompt suffix appended after the SFT template (matches build_prm_prompt)
_PRM_SUFFIX = "\nGenerated SQL Prefix Context:\n{prefix}\nThe next SQL clause to be generated is shown below. Please evaluate its logical correctness."

# CLAUSE_ORDER matches dataset/expand_prm_dataset.py
CLAUSE_ORDER = ["SELECT", "FROM", "JOIN", "WHERE", "GROUP_BY", "HAVING", "ORDER_BY", "LIMIT"]

# ── lazy singleton: load RM once per process ──────────────────────────────────
_rm_lock      = threading.Lock()
_rm_model     = None
_rm_tokenizer = None
_rm_device    = None


def _load_rm() -> None:
    global _rm_model, _rm_tokenizer, _rm_device
    with _rm_lock:
        if _rm_model is not None:
            return
        from transformers import AutoTokenizer
        from openrlhf.models import get_llm_for_sequence_regression

        print(f"[reward_func] Loading RM tokenizer from {_RM_PATH}", flush=True)
        _rm_tokenizer = AutoTokenizer.from_pretrained(_RM_PATH, trust_remote_code=True)

        print(f"[reward_func] Loading RM model from {_RM_PATH}", flush=True)
        _rm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_llm_for_sequence_regression(
            _RM_PATH,
            "reward",
            normalize_reward=False,
            attn_implementation="flash_attention_2",
            param_dtype="bf16",
            value_head_prefix="score",
            device_map=str(_rm_device),
        )
        model.eval()
        _rm_model = model
        print(f"[reward_func] RM loaded on {_rm_device}, vocab_size={_rm_tokenizer.vocab_size}",
              flush=True)


# ── SQL extraction ────────────────────────────────────────────────────────────

def _extract_sql(query: str) -> str:
    """Extract the SQL from <answer>```sql ... ```</answer> in actor output."""
    # Strategy 1: <answer> + ```sql block
    m = re.search(r"<answer>.*?```sql\s*(.*?)\s*```.*?</answer>", query, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Strategy 2: bare ```sql block
    m = re.search(r"```sql\s*(.*?)\s*```", query, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Strategy 3: last SELECT
    pos = query.upper().rfind("SELECT")
    if pos != -1:
        return query[pos:].strip()
    return ""


# ── chat-prompt parsing ───────────────────────────────────────────────────────

def _parse_chat_prompt(prompt: str):
    """
    Extract (schema_str, question, evidence) from the chat-template PPO prompt.

    Expected format (built by build_bird_ppo_dataset.py):
        <|im_start|>system
        ...
        Database Schema:
        <schema text>

        [External Knowledge:\n<evidence>]
        <|im_end|>
        <|im_start|>user
        <question>
        <|im_end|>
        <|im_start|>assistant
        <think>
    """
    # -- system content
    sys_m = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    system_content = sys_m.group(1) if sys_m else ""

    # schema: text after "Database Schema:\n" until end of system or "External Knowledge"
    schema_m = re.search(
        r"Database Schema:\n(.*?)(?=\n\nExternal Knowledge:|\Z)", system_content, re.DOTALL
    )
    schema_str = schema_m.group(1).strip() if schema_m else ""

    # evidence (optional)
    ev_m = re.search(r"External Knowledge:\n(.*?)(?=\Z)", system_content, re.DOTALL)
    evidence = ev_m.group(1).strip() if ev_m else ""

    # -- user content = question
    user_m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    question = user_m.group(1).strip() if user_m else ""

    return schema_str, question, evidence


# ── PRM prompt builder ────────────────────────────────────────────────────────

def _build_sft_prompt(schema_str: str, question: str) -> str:
    return _SFT_TMPL.format(db_details=schema_str, question=question).strip()


def _build_prm_input(sft_prompt: str, prefix: str, clause: str) -> str:
    """Full RM input = sft_prompt + prefix context header + clause text."""
    prm = sft_prompt + _PRM_SUFFIX.format(prefix=prefix if prefix else "(none yet)")
    return prm + "\n" + clause


# ── sqlglot clause decomposition ─────────────────────────────────────────────

def _decompose_sql(sql: str) -> List[tuple]:
    """
    Returns list of (clause_name, clause_text) in CLAUSE_ORDER.
    Falls back to [(\"SELECT\", sql)] if parsing fails.
    """
    try:
        import sqlglot
        from sqlglot import exp

        root = sqlglot.parse_one(sql, read="sqlite")

        def _select_root(node):
            if isinstance(node, exp.Select):
                return node
            for child in node.walk():
                if isinstance(child, exp.Select):
                    return child
            return None

        sroot = _select_root(root)
        if sroot is None:
            return [("SELECT", sql)]

        clauses: Dict[str, str] = {}

        sel = sroot.args.get("expressions") or []
        if sel:
            clauses["SELECT"] = "SELECT " + ", ".join(e.sql(dialect="sqlite") for e in sel)

        frm = sroot.args.get("from")
        if frm:
            clauses["FROM"] = frm.sql(dialect="sqlite")

        joins = sroot.args.get("joins") or []
        if joins:
            clauses["JOIN"] = " ".join(j.sql(dialect="sqlite") for j in joins)

        where = sroot.args.get("where")
        if where:
            clauses["WHERE"] = where.sql(dialect="sqlite")

        group = sroot.args.get("group")
        if group:
            clauses["GROUP_BY"] = group.sql(dialect="sqlite")

        having = sroot.args.get("having")
        if having:
            clauses["HAVING"] = having.sql(dialect="sqlite")

        order = sroot.args.get("order")
        if order:
            clauses["ORDER_BY"] = order.sql(dialect="sqlite")

        limit = sroot.args.get("limit")
        if limit:
            clauses["LIMIT"] = limit.sql(dialect="sqlite")

        result = [(n, clauses[n]) for n in CLAUSE_ORDER if n in clauses]
        return result if result else [("SELECT", sql)]

    except Exception:
        return [("SELECT", sql)]


# ── RM scoring ────────────────────────────────────────────────────────────────

def _score_texts(texts: List[str]) -> List[float]:
    """Run RM forward on a list of texts, return scalar rewards."""
    enc = _rm_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_RM_MAX_LEN,
    )
    input_ids      = enc["input_ids"].to(_rm_device)
    attention_mask = enc["attention_mask"].to(_rm_device)

    with torch.no_grad():
        rewards = _rm_model(input_ids, attention_mask)   # shape (B,)

    return rewards.squeeze(-1).float().cpu().tolist()


# ── aggregation ───────────────────────────────────────────────────────────────

def _aggregate(scores: List[float]) -> float:
    if not scores:
        return 0.0
    if _AGGREGATION == "min":
        return min(scores)
    if _AGGREGATION == "last":
        return scores[-1]
    return sum(scores) / len(scores)   # "mean"


# ── main entry point ──────────────────────────────────────────────────────────

def reward_func(
    queries: List[str],
    prompts: List[str],
    labels: List[str],
) -> Dict[str, Any]:
    """
    PRM-based reward: decompose SQL into clauses, score each with RM, aggregate.

    Called one sample at a time (batch_size=1) by OpenRLHF's
    SingleTurnAgentExecutor._fetch_rewards_via_func.
    """
    _load_rm()

    assert len(queries) == len(prompts), "queries/prompts length mismatch"

    all_rewards: List[float] = []
    all_clause_counts: List[int] = []

    for query, prompt in zip(queries, prompts):
        # 1. Extract SQL from actor output
        sql = _extract_sql(query)
        if not sql:
            all_rewards.append(0.0)
            all_clause_counts.append(0)
            continue

        # 2. Parse prompt to rebuild PRM-style sft_prompt
        schema_str, question, _evidence = _parse_chat_prompt(prompt)
        sft_prompt = _build_sft_prompt(schema_str, question)

        # 3. Decompose SQL into clauses
        clauses = _decompose_sql(sql)   # List[(name, text)]

        # 4. Build RM input for each clause
        rm_inputs: List[str] = []
        prefix_parts: List[str] = []
        for _name, clause_text in clauses:
            prefix = "\n".join(prefix_parts)
            rm_inputs.append(_build_prm_input(sft_prompt, prefix, clause_text))
            prefix_parts.append(clause_text)

        # 5. Score all clauses in one batch
        if rm_inputs:
            clause_scores = _score_texts(rm_inputs)
        else:
            clause_scores = [0.0]

        # 6. Aggregate
        reward = _aggregate(clause_scores)
        all_rewards.append(reward)
        all_clause_counts.append(len(clause_scores))

    # OpenRLHF expects scalar when batch_size=1
    reward_out = all_rewards[0] if len(all_rewards) == 1 else all_rewards
    score_out  = reward_out

    return {
        "rewards": reward_out,
        "scores":  score_out,
        "extra_logs": {
            "rm_reward_mean":  sum(all_rewards) / max(len(all_rewards), 1),
            "avg_clause_count": sum(all_clause_counts) / max(len(all_clause_counts), 1),
            "aggregation":     _AGGREGATION,
        },
    }
