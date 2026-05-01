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
import math
import datetime
import threading
import time
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
try:
    import pysqlite3 as sqlite3
except Exception:  # pragma: no cover - fallback
    import sqlite3

# ── repo root on sys.path so dataset/ subpackage is importable ────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_REPO_ROOT, os.path.join(_REPO_ROOT, "dataset")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── config via env vars ───────────────────────────────────────────────────────
_RM_PATH      = os.environ.get("REWARD_MODEL_PATH",
                               "/home/banlujie/RL-NL2SQL/ckpt/qwen3-0.6b-rm")
_RM_MAX_LEN   = int(os.environ.get("REWARD_MODEL_MAX_LEN", "2048"))
_REWARD_LOG_EVERY = int(os.environ.get("REWARD_FUNC_LOG_EVERY", "1"))

# Fallback shaping keeps a learning signal even when PRM clause scoring collapses.
_FORMAT_PENALTY_NO_SQL = float(os.environ.get("REWARD_FALLBACK_NO_SQL", "-0.15"))
_FORMAT_PENALTY_EMPTY_CLAUSE = float(os.environ.get("REWARD_FALLBACK_EMPTY_CLAUSE", "-0.05"))
_FORMAT_BONUS_ANSWER_SQL_FENCE = float(os.environ.get("REWARD_FALLBACK_ANSWER_SQL_FENCE", "0.005"))
_FORMAT_BONUS_SQL_FENCE = float(os.environ.get("REWARD_FALLBACK_SQL_FENCE", "0.003"))
_FORMAT_BONUS_SQL_KEYWORD = float(os.environ.get("REWARD_FALLBACK_SQL_KEYWORD", "0.00"))
_FORMAT_BONUS_HAS_ANSWER_BLOCK = float(os.environ.get("REWARD_FALLBACK_HAS_ANSWER_BLOCK", "0.002"))
_FORMAT_PENALTY_MISSING_ANSWER = float(os.environ.get("REWARD_FALLBACK_MISSING_ANSWER", "-0.12"))
_FORMAT_PENALTY_UNCLOSED_ANSWER = float(os.environ.get("REWARD_FALLBACK_UNCLOSED_ANSWER", "-0.20"))
_FORMAT_PENALTY_THINK_ONLY = float(os.environ.get("REWARD_FALLBACK_THINK_ONLY", "-0.10"))
_THINK_EMPTY_THRESHOLD = int(os.environ.get("REWARD_FALLBACK_THINK_EMPTY_THRESHOLD", "3"))
_THINK_EMPTY_PENALTY = float(os.environ.get("REWARD_FALLBACK_THINK_EMPTY_PENALTY", "-0.08"))
_THINK_WORD_BUDGET = int(os.environ.get("REWARD_FALLBACK_THINK_WORD_BUDGET", "120"))
_THINK_OVER_BUDGET_PENALTY = float(os.environ.get("REWARD_FALLBACK_THINK_OVER_BUDGET_PENALTY", "-0.06"))
_THINK_OVER_BUDGET_NO_ANSWER_PENALTY = float(
    os.environ.get("REWARD_FALLBACK_THINK_OVER_BUDGET_NO_ANSWER_PENALTY", "-0.30")
)
_REPEAT_NGRAM_SIZE = max(int(os.environ.get("REWARD_FALLBACK_REPEAT_NGRAM_SIZE", "12")), 2)
_REPEAT_RATIO_THRESHOLD = float(os.environ.get("REWARD_FALLBACK_REPEAT_RATIO_THRESHOLD", "0.20"))
_REPEAT_MIN_NGRAM_FREQ = max(int(os.environ.get("REWARD_FALLBACK_REPEAT_MIN_NGRAM_FREQ", "6")), 2)
_REPEAT_LOOP_PENALTY = float(os.environ.get("REWARD_FALLBACK_REPEAT_LOOP_PENALTY", "-0.25"))
_ALLOW_LOOSE_SELECT_FALLBACK = os.environ.get("REWARD_ALLOW_LOOSE_SELECT_FALLBACK", "0") == "1"
_REWARD_CLIP_LOW = float(os.environ.get("REWARD_FALLBACK_CLIP_LOW", "-1.0"))
_REWARD_CLIP_HIGH = float(os.environ.get("REWARD_FALLBACK_CLIP_HIGH", "1.0"))
_STRUCT_PENALTY_PARSE_FAIL = float(os.environ.get("REWARD_STRUCT_PENALTY_PARSE_FAIL", "-0.45"))
_STRUCT_PENALTY_COLUMN_WITHOUT_FROM = float(
    os.environ.get("REWARD_STRUCT_PENALTY_COLUMN_WITHOUT_FROM", "-0.35")
)
_STRUCT_PENALTY_JOIN_WITHOUT_FROM = float(
    os.environ.get("REWARD_STRUCT_PENALTY_JOIN_WITHOUT_FROM", "-0.45")
)
_STRUCT_PENALTY_UNKNOWN_ALIAS = float(os.environ.get("REWARD_STRUCT_PENALTY_UNKNOWN_ALIAS", "-0.20"))
_STRUCT_BONUS_HAS_FROM = float(os.environ.get("REWARD_STRUCT_BONUS_HAS_FROM", "0.03"))
_EXEC_PARSE_FAIL_REWARD = float(os.environ.get("REWARD_EXEC_PARSE_FAIL", "-1.0"))
_EXEC_MATCH_REWARD = float(os.environ.get("REWARD_EXEC_MATCH_REWARD", "1.0"))
_EXEC_TIMEOUT_REWARD = float(os.environ.get("REWARD_EXEC_TIMEOUT_REWARD", "-0.6"))
_EXEC_ERROR_REWARD = float(os.environ.get("REWARD_EXEC_ERROR_REWARD", "-0.4"))
_EXEC_TIMEOUT_SEC = float(os.environ.get("REWARD_EXEC_TIMEOUT_SEC", "5.0"))
_CLAUSE_EXEC_LAMBDA = float(os.environ.get("REWARD_CLAUSE_EXEC_LAMBDA", "0.1"))
_CLAUSE_MATCH_BONUS_MAX = float(os.environ.get("REWARD_CLAUSE_MATCH_BONUS_MAX", "0.5"))
_CLAUSE_SIGMOID_TEMPERATURE = float(os.environ.get("REWARD_CLAUSE_SIGMOID_TEMPERATURE", "1.0"))
_CLAUSE_SIGMOID_BIAS = float(os.environ.get("REWARD_CLAUSE_SIGMOID_BIAS", "0.5"))
_NONFINITE_REWARD_FALLBACK = float(os.environ.get("REWARD_NONFINITE_FALLBACK", "-1.0"))
_FORMAT_INVALID_REWARD = float(os.environ.get("REWARD_FORMAT_INVALID_REWARD", "-1.0"))
_ENABLE_EXEC_REWARD = os.environ.get("REWARD_ENABLE_EXEC", "1") == "1"
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
_reward_log_lock = threading.Lock()
_reward_call_idx = 0
_schema_to_db_path: Dict[str, str] = {}
_db_id_to_db_path: Dict[str, str] = {}
_table_columns_cache: Dict[str, Dict[str, set]] = {}
_schema_index_lock = threading.Lock()


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
        _rm_device = _resolve_rm_device()
        if _rm_device.type == "cuda":
            try:
                torch.cuda.set_device(_rm_device)
            except Exception:
                pass
        model = get_llm_for_sequence_regression(
            _RM_PATH,
            "reward",
            # Keep raw score-head outputs; this RM config produces non-finite values when normalized.
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

def _extract_generated_segment(query: str) -> str:
    """Best-effort isolate assistant generated segment to avoid prompt leakage."""
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
    """Extract SQL and return format metadata for fallback shaping."""
    generated = _extract_generated_segment(query)
    q_lower = generated.lower()
    has_think_open = "<think>" in q_lower
    has_think_close = "</think>" in q_lower
    has_answer_open = "<answer>" in q_lower
    has_answer_close = "</answer>" in q_lower
    has_answer_tag = has_answer_open and has_answer_close
    valid_format = bool(
        re.match(
            r"^\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$",
            generated,
            flags=re.IGNORECASE,
        )
    )
    has_answer_sql_fence = bool(
        re.search(r"<answer>.*?```sql\s*.*?\s*```.*?</answer>", generated, re.DOTALL | re.IGNORECASE)
    )
    has_sql_fence = bool(re.search(r"```sql\s*.*?\s*```", generated, re.DOTALL | re.IGNORECASE))
    think_word_count = 0
    think_text = ""
    think_repeat_ngram_ratio = 0.0
    think_repeat_max_ngram_freq = 0
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", generated, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_text = think_match.group(1)
        think_word_count = len(re.findall(r"\S+", think_text))
    elif has_think_open:
        open_pos = q_lower.rfind("<think>")
        think_tail = generated[open_pos + len("<think>") :]
        think_text = think_tail
        think_word_count = len(re.findall(r"\S+", think_text))

    think_tokens = re.findall(r"\w+", think_text.lower())
    if len(think_tokens) >= _REPEAT_NGRAM_SIZE:
        ngrams = [
            " ".join(think_tokens[i : i + _REPEAT_NGRAM_SIZE])
            for i in range(len(think_tokens) - _REPEAT_NGRAM_SIZE + 1)
        ]
        ngram_counts = Counter(ngrams)
        think_repeat_max_ngram_freq = max(ngram_counts.values())
        repeated_positions = sum(c for c in ngram_counts.values() if c > 1)
        think_repeat_ngram_ratio = repeated_positions / max(len(ngrams), 1)

    format_meta: Dict[str, Any] = {
        "has_think_open": has_think_open,
        "has_think_close": has_think_close,
        "has_answer_open": has_answer_open,
        "has_answer_close": has_answer_close,
        "has_answer_tag": has_answer_tag,
        "valid_format": valid_format,
        "has_answer_sql_fence": has_answer_sql_fence,
        "has_sql_fence": has_sql_fence,
        "think_word_count": think_word_count,
        "think_repeat_ngram_ratio": think_repeat_ngram_ratio,
        "think_repeat_max_ngram_freq": think_repeat_max_ngram_freq,
        "generated_chars": len(generated),
    }

    # Strategy 1: prefer SQL fences inside <answer>...</answer>, pick last non-empty.
    answer_blocks = re.findall(r"<answer>(.*?)</answer>", generated, re.DOTALL | re.IGNORECASE)
    for block in reversed(answer_blocks):
        sql_blocks = re.findall(r"```sql\s*(.*?)\s*```", block, re.DOTALL | re.IGNORECASE)
        for sql in reversed(sql_blocks):
            sql = sql.strip()
            if sql:
                return sql, {
                    **format_meta,
                }

    # Intentionally do not fall back to SQL found outside <answer>.
    # This prevents "think-only" outputs from receiving SQL rewards.
    return "", format_meta


def _extract_sql(query: str) -> str:
    sql, _ = _extract_sql_with_meta(query)
    return sql


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

    db_id_m = re.search(r"Database ID:\s*([^\n]+)", system_content, re.IGNORECASE)
    db_id = db_id_m.group(1).strip() if db_id_m else ""

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

    return schema_str, question, evidence, db_id


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
            db_path = ""
            for root in _BIRD_DB_ROOTS:
                cand = os.path.join(root, db_id, f"{db_id}.sqlite")
                if os.path.exists(cand):
                    db_path = cand
                    break
            if db_path:
                index[key] = db_path
    return index


def _build_db_id_index() -> Dict[str, str]:
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
            db_id = (item.get("db_id") or "").strip()
            if not db_id:
                continue
            if db_id in index and os.path.exists(index[db_id]):
                continue
            for root in _BIRD_DB_ROOTS:
                cand = os.path.join(root, db_id, f"{db_id}.sqlite")
                if os.path.exists(cand):
                    index[db_id] = cand
                    break
    return index


def _resolve_db_path(db_id: str, schema_str: str) -> Optional[str]:
    if not _ENABLE_EXEC_REWARD:
        return None
    with _schema_index_lock:
        if db_id:
            if not _db_id_to_db_path:
                _db_id_to_db_path.update(_build_db_id_index())
            if db_id in _db_id_to_db_path:
                return _db_id_to_db_path[db_id]
        key = _normalize_schema_text(schema_str)
        if not key:
            return None
        if not _schema_to_db_path:
            _schema_to_db_path.update(_build_schema_db_index())
        return _schema_to_db_path.get(key)


def _run_sql_with_timeout(db_path: str, sql: str, timeout_sec: float) -> Tuple[Optional[List[Tuple[Any, ...]]], Optional[str]]:
    start = time.time()
    conn = sqlite3.connect(db_path)
    try:
        if hasattr(conn, "set_progress_handler"):
            def _progress():
                if time.time() - start > timeout_sec:
                    return 1
                return 0
            conn.set_progress_handler(_progress, 10000)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return rows, None
    except Exception as e:
        msg = str(e).lower()
        if "interrupted" in msg:
            return None, "timeout"
        return None, f"error:{type(e).__name__}"
    finally:
        conn.close()


def _rewrite_mmddyyyy_to_iso(sql: str) -> str:
    """
    Rewrite date string literals like '9/9/2019' or '9-9-2019' to '2019-09-09'.
    Only rewrites valid calendar dates.
    """
    if not sql:
        return sql

    def _repl(match: re.Match) -> str:
        m = int(match.group(1))
        d = int(match.group(2))
        y = int(match.group(3))
        try:
            _ = datetime.date(y, m, d)
        except ValueError:
            return match.group(0)
        return f"'{y:04d}-{m:02d}-{d:02d}'"

    out = re.sub(r"'(\d{1,2})/(\d{1,2})/(\d{4})'", _repl, sql)
    out = re.sub(r"'(\d{1,2})-(\d{1,2})-(\d{4})'", _repl, out)
    return out


def _quote_ident(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def _get_table_columns(db_path: str, table_name: str) -> set:
    db_cache = _table_columns_cache.setdefault(db_path, {})
    if table_name in db_cache:
        return db_cache[table_name]
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({_quote_ident(table_name)})")
        rows = cur.fetchall()
        cols = {str(r[1]).lower() for r in rows if len(r) > 1 and r[1] is not None}
        db_cache[table_name] = cols
        return cols
    except Exception:
        return set()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _repair_unknown_alias(sql: str, db_path: Optional[str]) -> str:
    """
    Repair alias references like m.col where alias m is never defined.
    If all referenced columns of the unknown alias uniquely map to one table in the query,
    replace alias. with table.
    """
    if not sql or not db_path or not os.path.exists(db_path):
        return sql

    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return sql

    try:
        root = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return sql

    tables: List[str] = []
    known_refs = set()
    for t in root.find_all(exp.Table):
        table_name = (t.name or "").strip()
        if not table_name:
            continue
        alias_name = (t.alias_or_name or "").strip()
        tables.append(table_name)
        known_refs.add(table_name.lower())
        if alias_name:
            known_refs.add(alias_name.lower())

    if not tables:
        return sql

    unknown_alias_cols: Dict[str, set] = {}
    for col in root.find_all(exp.Column):
        alias = (col.table or "").strip()
        col_name = (col.name or "").strip()
        if not alias or not col_name:
            continue
        if alias.lower() in known_refs:
            continue
        unknown_alias_cols.setdefault(alias.lower(), set()).add(col_name.lower())

    if not unknown_alias_cols:
        return sql

    table_cols = {t: _get_table_columns(db_path, t) for t in tables}
    alias_to_table: Dict[str, str] = {}
    for alias, cols in unknown_alias_cols.items():
        candidates = [t for t in tables if cols.issubset(table_cols.get(t, set()))]
        if len(candidates) != 1:
            return sql
        alias_to_table[alias] = candidates[0]

    repaired = sql
    for alias, table_name in alias_to_table.items():
        repaired = re.sub(
            rf"(?<![A-Za-z0-9_`]){re.escape(alias)}\s*\.",
            f"{table_name}.",
            repaired,
            flags=re.IGNORECASE,
        )
    return repaired


def _pred_sql_candidates(pred_sql: str) -> List[str]:
    candidates: List[str] = []
    if pred_sql:
        staged = [pred_sql]
        normalized = _rewrite_mmddyyyy_to_iso(pred_sql)
        if normalized != pred_sql:
            staged.append(normalized)

        seen = set()
        for s in staged:
            if s not in seen:
                seen.add(s)
                candidates.append(s)
    return candidates


def _execute_match_reward(db_path: Optional[str], pred_sql: str, gt_sql: str) -> Tuple[float, str]:
    if not _ENABLE_EXEC_REWARD:
        return -1.0, "exec_disabled"
    if not db_path or not os.path.exists(db_path):
        return -1.0, "db_not_found"

    gt_rows, gt_err = _run_sql_with_timeout(db_path, gt_sql, _EXEC_TIMEOUT_SEC)
    if gt_err is not None:
        return -1.0, "gt_exec_error"

    had_pred_success = False
    base_candidates = _pred_sql_candidates(pred_sql)
    for cand_sql in base_candidates:
        pred_rows, pred_err = _run_sql_with_timeout(db_path, cand_sql, _EXEC_TIMEOUT_SEC)
        if pred_err is not None:
            continue
        had_pred_success = True
        if pred_rows == gt_rows:
            return _EXEC_MATCH_REWARD, "exec_match"
    if had_pred_success:
        return -1.0, "exec_mismatch"

    repaired_candidates: List[str] = []
    seen = set(base_candidates)
    for cand_sql in base_candidates:
        repaired = _repair_unknown_alias(cand_sql, db_path)
        if repaired != cand_sql and repaired not in seen:
            seen.add(repaired)
            repaired_candidates.append(repaired)
    for cand_sql in repaired_candidates:
        pred_rows, pred_err = _run_sql_with_timeout(db_path, cand_sql, _EXEC_TIMEOUT_SEC)
        if pred_err is not None:
            continue
        if pred_rows == gt_rows:
            return _EXEC_MATCH_REWARD, "exec_match"
        had_pred_success = True

    if not had_pred_success:
        return -1.0, "pred_exec_error"
    return -1.0, "exec_mismatch"


def _extract_gt_sql(label: Any) -> str:
    if isinstance(label, str):
        return label.strip()
    if isinstance(label, dict):
        for key in ("sql", "SQL", "gold_sql", "label", "answer"):
            value = label.get(key)
            if value:
                return str(value).strip()
    return ""


# ── PRM prompt builder ────────────────────────────────────────────────────────

def _build_sft_prompt(schema_str: str, question: str) -> str:
    return _SFT_TMPL.format(db_details=schema_str, question=question).strip()


def _build_prm_input(sft_prompt: str, prefix: str, clause: str) -> str:
    """Full RM input = sft_prompt + prefix context header + clause text."""
    prm = sft_prompt + _PRM_SUFFIX.format(prefix=prefix if prefix else "(none yet)")
    return prm + "\n" + clause


# ── sqlglot clause decomposition ─────────────────────────────────────────────

def _parse_sql_ast(sql: str) -> Tuple[Optional[Any], Optional[Any], bool]:
    """Parse SQL once and reuse AST for decomposition + structural checks."""
    try:
        import sqlglot
        from sqlglot import exp

        return sqlglot.parse_one(sql, read="sqlite"), exp, False
    except Exception:
        return None, None, True


def _decompose_sql(sql: str, root: Optional[Any] = None, exp: Optional[Any] = None, parse_failed: bool = False) -> List[tuple]:
    """
    Returns list of (clause_name, clause_text) in CLAUSE_ORDER.
    Falls back to [(\"SELECT\", sql)] if parsing fails.
    """
    try:
        if parse_failed:
            return [("SELECT", sql)]
        if root is None or exp is None:
            root, exp, parse_failed = _parse_sql_ast(sql)
            if parse_failed or root is None or exp is None:
                return [("SELECT", sql)]

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


def _normalize_clause_text(clause_text: str) -> str:
    return re.sub(r"\s+", " ", (clause_text or "").strip().lower())


def _clause_match_bonus(pred_clauses: List[Tuple[str, str]], label_sql: str) -> Tuple[float, float]:
    """
    Give explicit positive reward to clauses that match ground-truth clauses.
    Returns (bonus, match_ratio).
    """
    if not pred_clauses or not label_sql or _CLAUSE_MATCH_BONUS_MAX <= 0.0:
        return 0.0, 0.0

    gold_root, gold_exp, gold_parse_failed = _parse_sql_ast(label_sql)
    if gold_parse_failed:
        return 0.0, 0.0
    gold_clauses = _decompose_sql(label_sql, root=gold_root, exp=gold_exp, parse_failed=False)
    if not gold_clauses:
        return 0.0, 0.0

    gold_map = {name: _normalize_clause_text(text) for name, text in gold_clauses}
    matched = 0
    total = 0
    for name, text in pred_clauses:
        total += 1
        if gold_map.get(name, "") == _normalize_clause_text(text):
            matched += 1
    if total <= 0:
        return 0.0, 0.0

    ratio = matched / total
    return _CLAUSE_MATCH_BONUS_MAX * ratio, ratio


def _structural_sql_penalty(
    sql: str,
    root: Optional[Any] = None,
    exp: Optional[Any] = None,
    parse_failed: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Penalize reward-hacking patterns that are syntactically shaped but structurally invalid.
    """
    diagnostics = {
        "struct_parse_failed": 0.0,
        "struct_has_columns_no_from": 0.0,
        "struct_join_without_from": 0.0,
        "struct_unknown_alias_ref_count": 0.0,
    }
    if not sql:
        return 0.0, diagnostics

    if parse_failed:
        diagnostics["struct_parse_failed"] = 1.0
        return _STRUCT_PENALTY_PARSE_FAIL, diagnostics
    if root is None or exp is None:
        root, exp, parse_failed = _parse_sql_ast(sql)
        if parse_failed or root is None or exp is None:
            diagnostics["struct_parse_failed"] = 1.0
            return _STRUCT_PENALTY_PARSE_FAIL, diagnostics

    select_root = None
    if isinstance(root, exp.Select):
        select_root = root
    else:
        for node in root.walk():
            if isinstance(node, exp.Select):
                select_root = node
                break

    if select_root is None:
        return 0.0, diagnostics

    penalty = 0.0
    has_from = bool(select_root.args.get("from"))
    has_join = bool(select_root.args.get("joins") or [])
    columns = list(select_root.find_all(exp.Column))
    if has_from:
        penalty += _STRUCT_BONUS_HAS_FROM
    if has_join and not has_from:
        diagnostics["struct_join_without_from"] = 1.0
        penalty += _STRUCT_PENALTY_JOIN_WITHOUT_FROM
    if columns and not has_from:
        diagnostics["struct_has_columns_no_from"] = 1.0
        penalty += _STRUCT_PENALTY_COLUMN_WITHOUT_FROM

    known_aliases = set()
    for table in select_root.find_all(exp.Table):
        table_name = (table.name or "").lower()
        if table_name:
            known_aliases.add(table_name)
        alias_name = (table.alias_or_name or "").lower()
        if alias_name:
            known_aliases.add(alias_name)

    unknown_alias_ref_count = 0
    for col in columns:
        table_name = (col.table or "").lower()
        if table_name and table_name not in known_aliases:
            unknown_alias_ref_count += 1
    if unknown_alias_ref_count > 0:
        diagnostics["struct_unknown_alias_ref_count"] = float(unknown_alias_ref_count)
        penalty += _STRUCT_PENALTY_UNKNOWN_ALIAS * min(unknown_alias_ref_count, 3)

    return penalty, diagnostics


# ── RM scoring ────────────────────────────────────────────────────────────────

def _score_texts(texts: List[str]) -> List[float]:
    """Run RM forward on a list of texts, return scalar rewards."""
    _load_rm()
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

    # Keep return type stable as List[float] even for a single-element batch.
    raw = rewards.float().reshape(-1).cpu().tolist()
    return [v if math.isfinite(v) else _NONFINITE_REWARD_FALLBACK for v in raw]


def _signed_sigmoid_score(raw_score: float) -> float:
    """
    Map raw RM score to [-1, 1] so positive means better and negative means worse.
    """
    if not math.isfinite(raw_score):
        return _NONFINITE_REWARD_FALLBACK
    z = (_CLAUSE_SIGMOID_TEMPERATURE * float(raw_score)) + _CLAUSE_SIGMOID_BIAS
    z = max(min(z, 40.0), -40.0)
    s = 1.0 / (1.0 + math.exp(-z))
    return (2.0 * s) - 1.0


# ── aggregation ───────────────────────────────────────────────────────────────

def _aggregate(scores: List[float]) -> float:
    if isinstance(scores, (int, float)):
        return float(scores)
    if not scores:
        return 0.0
    # Online reward uses mean clause score aggregation.
    out = sum(scores) / len(scores)
    return out if math.isfinite(out) else _NONFINITE_REWARD_FALLBACK


def _format_fallback_reward(sql: str, format_meta: Dict[str, Any], clause_count: int) -> float:
    """
    Small shaping reward to avoid all-zero training signals when formatting degrades.
    """
    reward = 0.0
    has_answer_open = bool(format_meta.get("has_answer_open", False))
    has_answer_tag = bool(format_meta.get("has_answer_tag", False))
    has_think_open = bool(format_meta.get("has_think_open", False))

    if has_answer_tag:
        reward += _FORMAT_BONUS_HAS_ANSWER_BLOCK
    elif has_answer_open:
        reward += _FORMAT_PENALTY_UNCLOSED_ANSWER
    else:
        reward += _FORMAT_PENALTY_MISSING_ANSWER

    if has_think_open and not has_answer_open:
        reward += _FORMAT_PENALTY_THINK_ONLY

    think_word_count = int(format_meta.get("think_word_count", 0))
    has_think_open = bool(format_meta.get("has_think_open", False))
    has_think_close = bool(format_meta.get("has_think_close", False))
    if has_think_open and has_think_close and think_word_count < _THINK_EMPTY_THRESHOLD:
        reward += _THINK_EMPTY_PENALTY
    has_answer_close = bool(format_meta.get("has_answer_close", False))
    has_answer_complete = has_answer_open and has_answer_close
    if think_word_count > _THINK_WORD_BUDGET:
        reward += _THINK_OVER_BUDGET_PENALTY
        if not has_answer_complete:
            reward += _THINK_OVER_BUDGET_NO_ANSWER_PENALTY

    think_repeat_ngram_ratio = float(format_meta.get("think_repeat_ngram_ratio", 0.0))
    think_repeat_max_ngram_freq = int(format_meta.get("think_repeat_max_ngram_freq", 0))
    if (
        think_repeat_ngram_ratio >= _REPEAT_RATIO_THRESHOLD
        or think_repeat_max_ngram_freq >= _REPEAT_MIN_NGRAM_FREQ
    ):
        reward += _REPEAT_LOOP_PENALTY

    if not sql:
        return reward + _FORMAT_PENALTY_NO_SQL

    if format_meta.get("has_answer_sql_fence", False):
        reward += _FORMAT_BONUS_ANSWER_SQL_FENCE
    elif format_meta.get("has_sql_fence", False):
        reward += _FORMAT_BONUS_SQL_FENCE

    sql_upper = sql.lstrip().upper()
    if sql_upper.startswith(("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")):
        reward += _FORMAT_BONUS_SQL_KEYWORD

    if clause_count <= 0:
        reward += _FORMAT_PENALTY_EMPTY_CLAUSE
    return reward


def _clip_reward(x: float) -> float:
    return max(_REWARD_CLIP_LOW, min(_REWARD_CLIP_HIGH, x))


def _is_think_over_budget_without_answer(format_meta: Dict[str, Any]) -> bool:
    think_word_count = int(format_meta.get("think_word_count", 0))
    has_answer_open = bool(format_meta.get("has_answer_open", False))
    has_answer_close = bool(format_meta.get("has_answer_close", False))
    return think_word_count > _THINK_WORD_BUDGET and not (has_answer_open and has_answer_close)


def _is_repeat_loop(format_meta: Dict[str, Any]) -> bool:
    think_repeat_ngram_ratio = float(format_meta.get("think_repeat_ngram_ratio", 0.0))
    think_repeat_max_ngram_freq = int(format_meta.get("think_repeat_max_ngram_freq", 0))
    return (
        think_repeat_ngram_ratio >= _REPEAT_RATIO_THRESHOLD
        or think_repeat_max_ngram_freq >= _REPEAT_MIN_NGRAM_FREQ
    )


def _log_batch_metrics(
    batch_size: int,
    sql_extract_success_rate: float,
    clause_nonempty_rate: float,
    nonzero_reward_rate: float,
    rm_reward_raw_mean: float,
    fallback_reward_mean: float,
    final_reward_mean: float,
) -> None:
    global _reward_call_idx
    with _reward_log_lock:
        _reward_call_idx += 1
        if _REWARD_LOG_EVERY > 0 and (_reward_call_idx % _REWARD_LOG_EVERY == 0):
            print(
                (
                    f"[reward_func] batch={_reward_call_idx} size={batch_size} "
                    f"extract_rate={sql_extract_success_rate:.3f} "
                    f"clause_nonempty_rate={clause_nonempty_rate:.3f} "
                    f"nonzero_reward_rate={nonzero_reward_rate:.3f} "
                    f"rm_raw_mean={rm_reward_raw_mean:.4f} "
                    f"fallback_mean={fallback_reward_mean:.4f} "
                    f"final_mean={final_reward_mean:.4f}"
                ),
                flush=True,
            )


# ── main entry point ──────────────────────────────────────────────────────────

def reward_func(
    queries: List[str],
    prompts: List[str],
    labels: List[str],
) -> Dict[str, Any]:
    """
    Reward:
        R = I_format * (lambda * r_clause + (1 - lambda) * r_exec)
    where:
        I_format in {0,1} is SQL parse validity indicator from sqlglot,
        r_clause is Clause-PRM score (aggregated over predicted clauses),
        r_exec in {-1,1} is execution reward (+1 for exact-match, else -1).
    """
    assert len(queries) == len(prompts) == len(labels), "queries/prompts/labels length mismatch"

    lam = max(0.0, min(1.0, _CLAUSE_EXEC_LAMBDA))
    all_rewards: List[float] = []
    all_format_flags: List[float] = []
    all_clause_rewards: List[float] = []
    all_clause_rewards_raw: List[float] = []
    all_exec_rewards: List[float] = []
    all_clause_counts: List[int] = []
    all_clause_scores: List[List[float]] = []
    all_clause_scores_raw: List[List[float]] = []
    all_clause_texts: List[List[str]] = []
    all_clause_names: List[List[str]] = []
    all_exec_statuses: List[str] = []

    for query, prompt, label in zip(queries, prompts, labels):
        sql, _format_meta = _extract_sql_with_meta(query)
        root, exp, parse_failed = _parse_sql_ast(sql) if sql else (None, None, True)
        format_indicator = 0.0 if parse_failed else 1.0
        all_format_flags.append(format_indicator)

        # sqlglot parse failure receives explicit penalty to avoid think-only collapse.
        if format_indicator == 0.0:
            all_clause_rewards_raw.append(0.0)
            all_clause_rewards.append(0.0)
            all_exec_rewards.append(-1.0)
            all_clause_counts.append(0)
            all_clause_scores.append([])
            all_clause_scores_raw.append([])
            all_clause_texts.append([])
            all_clause_names.append([])
            all_exec_statuses.append("format_invalid")
            all_rewards.append(_FORMAT_INVALID_REWARD)
            continue

        schema_str, question, _evidence, db_id = _parse_chat_prompt(prompt)
        db_path = _resolve_db_path(db_id, schema_str)
        gt_sql = _extract_gt_sql(label)

        # Clause-PRM reward
        clauses = _decompose_sql(sql, root=root, exp=exp, parse_failed=False) if sql else []
        sft_prompt = _build_sft_prompt(schema_str, question)
        rm_inputs: List[str] = []
        prefix_parts: List[str] = []
        for _name, clause_text in clauses:
            prefix = "\n".join(prefix_parts)
            rm_inputs.append(_build_prm_input(sft_prompt, prefix, clause_text))
            prefix_parts.append(clause_text)
        clause_scores_raw = _score_texts(rm_inputs) if rm_inputs else [0.0]
        clause_scores = [_signed_sigmoid_score(s) for s in clause_scores_raw]
        r_clause_raw = float(_aggregate(clause_scores_raw))
        r_clause = float(_aggregate(clause_scores))
        if not math.isfinite(r_clause_raw):
            r_clause_raw = _NONFINITE_REWARD_FALLBACK
        if not math.isfinite(r_clause):
            r_clause = _NONFINITE_REWARD_FALLBACK

        # Binary execution reward
        r_exec, exec_status = _execute_match_reward(db_path, sql, gt_sql)
        r_exec = float(r_exec)

        if exec_status == "exec_match":
            reward = 1.0
        else:
            reward = format_indicator * (lam * r_clause + (1.0 - lam) * r_exec)
        reward = max(-1.0, min(1.0, reward))
        if not math.isfinite(reward):
            reward = _NONFINITE_REWARD_FALLBACK
        all_clause_rewards_raw.append(r_clause_raw)
        all_clause_rewards.append(r_clause)
        all_exec_rewards.append(r_exec)
        all_clause_counts.append(len(clauses))
        all_clause_scores.append([float(s) for s in clause_scores])
        all_clause_scores_raw.append([float(s) for s in clause_scores_raw])
        all_clause_texts.append([c for _n, c in clauses])
        all_clause_names.append([n for n, _c in clauses])
        all_exec_statuses.append(exec_status)
        all_rewards.append(float(reward))

    reward_out = all_rewards[0] if len(all_rewards) == 1 else all_rewards
    score_out = reward_out
    clause_scores_out = all_clause_scores[0] if len(all_clause_scores) == 1 else all_clause_scores
    clause_scores_raw_out = all_clause_scores_raw[0] if len(all_clause_scores_raw) == 1 else all_clause_scores_raw
    clause_texts_out = all_clause_texts[0] if len(all_clause_texts) == 1 else all_clause_texts
    clause_names_out = all_clause_names[0] if len(all_clause_names) == 1 else all_clause_names

    return {
        "rewards": reward_out,
        "scores": score_out,
        "extra_logs": {
            "reward_formula": "I_format * (lambda * r_clause + (1-lambda) * r_exec)",
            "lambda_clause_exec": lam,
            "format_valid_rate": sum(all_format_flags) / max(len(all_format_flags), 1),
            "clause_reward_raw_mean": sum(all_clause_rewards_raw) / max(len(all_clause_rewards_raw), 1),
            "clause_reward_mean": sum(all_clause_rewards) / max(len(all_clause_rewards), 1),
            "exec_reward_mean": sum(all_exec_rewards) / max(len(all_exec_rewards), 1),
            "final_reward_mean": sum(all_rewards) / max(len(all_rewards), 1),
            "exec_match_rate": sum(1.0 if s == "exec_match" else 0.0 for s in all_exec_statuses)
            / max(len(all_exec_statuses), 1),
            "avg_clause_count": sum(all_clause_counts) / max(len(all_clause_counts), 1),
            "exec_statuses": all_exec_statuses[0] if len(all_exec_statuses) == 1 else all_exec_statuses,
            "clause_scores": clause_scores_out,
            "clause_scores_raw": clause_scores_raw_out,
            "clause_texts": clause_texts_out,
            "clause_names": clause_names_out,
        },
    }
