import argparse
import copy
import math
import json
import os
import random
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm
import sqlglot
from sqlglot import exp

from utils import execute_sql_with_timeout, extract_sql_from_output


CLAUSE_ORDER = [
    "SELECT",
    "FROM",
    "JOIN",
    "WHERE",
    "GROUP_BY",
    "HAVING",
    "ORDER_BY",
    "LIMIT",
]

CLAUSE_PRIORITY = {name: idx for idx, name in enumerate(CLAUSE_ORDER)}


@dataclass
class EvalSample:
    question_id: int
    db_id: str
    prompt: str
    gt_sql: str
    pred_sql: str


@dataclass
class NodeMeta:
    preorder_idx: int
    path: Tuple[str, ...]
    node: exp.Expression
    clause: str


@dataclass
class BuildStats:
    total: int = 0
    pred_correct: int = 0
    pred_wrong: int = 0
    pred_empty: int = 0
    fix_success: int = 0
    inject_success: int = 0
    inject_from_gt_success: int = 0
    parse_fail: int = 0


def normalize_sql_text(sql: str) -> str:
    if not isinstance(sql, str):
        return ""
    sql = sql.strip()
    if sql.startswith("```sql") and sql.endswith("```"):
        sql = sql[6:-3].strip()
    return sql


def robust_extract_sql(raw_sql: str) -> str:
    cleaned = normalize_sql_text(raw_sql)
    extracted = extract_sql_from_output(cleaned)
    if extracted:
        return normalize_sql_text(extracted)
    return cleaned


def db_path_for(db_root: str, db_id: str) -> str:
    return os.path.join(db_root, db_id, f"{db_id}.sqlite")


@lru_cache(maxsize=200000)
def _execute_frozen_result_cached(db_path: str, sql: str, timeout_seconds: int = 5):
    return execute_sql_with_timeout(db_path, sql, timeout_seconds=timeout_seconds)


def execute_frozen_result(db_path: str, sql: str, timeout_seconds: int = 5):
    return _execute_frozen_result_cached(db_path, sql, timeout_seconds)


@lru_cache(maxsize=200000)
def result_equivalent(db_path: str, pred_sql: str, gt_sql: str, timeout_seconds: int = 5) -> Tuple[bool, str, str]:
    pred_res = execute_frozen_result(db_path, pred_sql, timeout_seconds)
    gt_res = execute_frozen_result(db_path, gt_sql, timeout_seconds)

    pred_err = isinstance(pred_res, str) and pred_res.startswith("Error")
    gt_err = isinstance(gt_res, str) and gt_res.startswith("Error")

    if gt_err:
        return False, "gt_error", str(gt_res)
    if pred_err:
        return False, "pred_error", str(pred_res)

    if pred_res == gt_res:
        return True, "equivalent", ""
    return False, "different_result", ""


def is_timeout_error(reason: str, info: str) -> bool:
    if reason not in {"gt_error", "pred_error"}:
        return False
    return "timed out" in str(info).lower() or "timeout" in str(info).lower()


def parse_sql(sql: str) -> Optional[exp.Expression]:
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None


def node_kind(node: exp.Expression) -> str:
    return type(node).__name__


def classify_clause(node: exp.Expression) -> str:
    if isinstance(node, exp.Select):
        return "SELECT"
    if node.find_ancestor(exp.Where):
        return "WHERE"
    if node.find_ancestor(exp.Having):
        return "HAVING"
    if node.find_ancestor(exp.Group):
        return "GROUP_BY"
    if node.find_ancestor(exp.Order):
        return "ORDER_BY"
    if node.find_ancestor(exp.Limit):
        return "LIMIT"
    if node.find_ancestor(exp.Join):
        return "JOIN"
    if node.find_ancestor(exp.From):
        return "FROM"
    if isinstance(node, exp.Join):
        return "JOIN"
    if isinstance(node, exp.From):
        return "FROM"
    if isinstance(node, exp.Where):
        return "WHERE"
    if isinstance(node, exp.Group):
        return "GROUP_BY"
    if isinstance(node, exp.Having):
        return "HAVING"
    if isinstance(node, exp.Order):
        return "ORDER_BY"
    if isinstance(node, exp.Limit):
        return "LIMIT"
    return "SELECT"


def _tokenize_sql(sql: str) -> List[str]:
    # A lightweight SQL tokenizer for span anchoring.
    import re

    pattern = r"(?:'[^']*'|\"[^\"]*\"|`[^`]*`|\w+|>=|<=|<>|!=|==|[-+*/%=<>()\[\],.;])"
    return re.findall(pattern, sql)


def _find_subsequence(haystack: List[str], needle: List[str], start: int = 0) -> Optional[Tuple[int, int]]:
    if not needle:
        return None
    n = len(needle)
    for i in range(max(0, start), len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i, i + n
    return None


def build_node_to_token_spans(sql: str, metas: List[NodeMeta]) -> Dict[str, List[int]]:
    sql_tokens = _tokenize_sql(sql)
    spans: Dict[str, List[int]] = {}
    cursor = 0

    for m in metas:
        node_sql = m.node.sql(dialect="sqlite")
        node_tokens = _tokenize_sql(node_sql)
        found = _find_subsequence(sql_tokens, node_tokens, start=cursor)
        if found is None:
            found = _find_subsequence(sql_tokens, node_tokens, start=0)
        if found is None:
            spans[f"node_idx_{m.preorder_idx}"] = []
            continue

        st, ed = found
        spans[f"node_idx_{m.preorder_idx}"] = list(range(st, ed))
        cursor = st

    return spans


def _is_ancestor(a_idx: int, b_idx: int, parents: Dict[int, Optional[int]]) -> bool:
    cur = b_idx
    while cur is not None and cur in parents:
        if cur == a_idx:
            return True
        cur = parents[cur]
    return a_idx == b_idx


def _ancestor_distance(ancestor_idx: int, descendant_idx: int, parents: Dict[int, Optional[int]]) -> Optional[int]:
    if ancestor_idx == descendant_idx:
        return 0
    dist = 0
    cur = descendant_idx
    while cur is not None and cur in parents:
        if cur == ancestor_idx:
            return dist
        cur = parents[cur]
        dist += 1
    return None


def _build_depth_map(parents: Dict[int, Optional[int]]) -> Dict[int, int]:
    depth: Dict[int, int] = {}

    def _depth_of(idx: int) -> int:
        if idx in depth:
            return depth[idx]
        p = parents.get(idx)
        if p is None:
            depth[idx] = 0
            return 0
        d = _depth_of(p) + 1
        depth[idx] = d
        return d

    for k in parents.keys():
        _depth_of(k)
    return depth


def _ancestor_chain(idx: int, parents: Dict[int, Optional[int]]) -> List[int]:
    chain = [idx]
    cur = idx
    while cur in parents and parents[cur] is not None:
        cur = parents[cur]
        chain.append(cur)
    return chain


def _lowest_common_ancestor(a_idx: int, b_idx: int, parents: Dict[int, Optional[int]]) -> Optional[int]:
    a_chain = _ancestor_chain(a_idx, parents)
    b_set = set(_ancestor_chain(b_idx, parents))
    for n in a_chain:
        if n in b_set:
            return n
    return None


def _causal_penalty_lca(
    t_hat_idx: int,
    t_star_idx: int,
    parents: Dict[int, Optional[int]],
    depths: Dict[int, int],
    alpha: float,
    sigma: float,
) -> float:
    # d-separation gating: if LCA is neither endpoint, branches are parallel -> no penalty.
    lca = _lowest_common_ancestor(t_hat_idx, t_star_idx, parents)
    if lca is None:
        return 0.0
    if lca != t_hat_idx and lca != t_star_idx:
        return 0.0

    d = abs(depths.get(t_hat_idx, 0) - depths.get(t_star_idx, 0))
    return float(-alpha * math.exp(-((d ** 2) / (2.0 * (sigma ** 2)))))


def _clause_expression_map(root: exp.Expression) -> List[Tuple[str, str]]:
    clauses: List[Tuple[str, str]] = []

    select_exprs = root.args.get("expressions") or []
    if select_exprs:
        select_sql = ", ".join(expr.sql(dialect="sqlite") for expr in select_exprs)
        clauses.append(("SELECT", f"SELECT {select_sql}"))

    from_expr = root.args.get("from")
    if isinstance(from_expr, exp.Expression):
        clauses.append(("FROM", from_expr.sql(dialect="sqlite")))

    joins = root.args.get("joins") or []
    for j in joins:
        if isinstance(j, exp.Expression):
            clauses.append(("JOIN", j.sql(dialect="sqlite")))

    where_expr = root.args.get("where")
    if isinstance(where_expr, exp.Expression):
        clauses.append(("WHERE", where_expr.sql(dialect="sqlite")))

    group_expr = root.args.get("group")
    if isinstance(group_expr, exp.Expression):
        clauses.append(("GROUP_BY", group_expr.sql(dialect="sqlite")))

    having_expr = root.args.get("having")
    if isinstance(having_expr, exp.Expression):
        clauses.append(("HAVING", having_expr.sql(dialect="sqlite")))

    order_expr = root.args.get("order")
    if isinstance(order_expr, exp.Expression):
        clauses.append(("ORDER_BY", order_expr.sql(dialect="sqlite")))

    limit_expr = root.args.get("limit")
    if isinstance(limit_expr, exp.Expression):
        clauses.append(("LIMIT", limit_expr.sql(dialect="sqlite")))

    return clauses


def build_causal_reward_annotations(
    sql: str,
    t_star_idx: int,
    alpha: float,
    sigma: float,
) -> Dict:
    root = parse_sql(sql)
    if root is None:
        return {
            "token_rewards": [],
            "node_to_token_spans": {},
            "macro_clauses": [],
            "clause_rewards": [],
        }

    metas = build_preorder_with_path(root)
    path_to_idx = {m.path: m.preorder_idx for m in metas}

    # Parent idx map for ancestor/descendant checks.
    parents: Dict[int, Optional[int]] = {}
    for m in metas:
        if len(m.path) == 0:
            parents[m.preorder_idx] = None
        else:
            parent_path = m.path[:-1]
            parents[m.preorder_idx] = path_to_idx.get(parent_path)

    if t_star_idx < 0 or t_star_idx >= len(metas):
        t_star_idx = min(max(0, t_star_idx), max(0, len(metas) - 1))

    depths = _build_depth_map(parents)

    # Node-level rewards with LCA-based d-separation masking.
    node_rewards: List[float] = []
    for m in metas:
        i = m.preorder_idx
        # Keep root neutral to avoid leaking global penalty into unrelated clause scores.
        if i == 0:
            node_rewards.append(0.0)
            continue

        score = _causal_penalty_lca(
            t_hat_idx=i,
            t_star_idx=t_star_idx,
            parents=parents,
            depths=depths,
            alpha=alpha,
            sigma=sigma,
        )
        node_rewards.append(score)

    node_to_token_spans = build_node_to_token_spans(sql, metas)
    sql_tokens = _tokenize_sql(sql)
    token_rewards = [0.0] * len(sql_tokens)

    for m, r in zip(metas, node_rewards):
        key = f"node_idx_{m.preorder_idx}"
        for tid in node_to_token_spans.get(key, []):
            if 0 <= tid < len(token_rewards):
                # Keep the most severe penalty when multiple AST nodes overlap this token.
                token_rewards[tid] = min(token_rewards[tid], r)

    clause_rewards_map: Dict[str, List[float]] = {}
    for m, r in zip(metas, node_rewards):
        if m.clause == "ROOT":
            continue
        clause_rewards_map.setdefault(m.clause, []).append(r)

    clause_items = _clause_expression_map(root)
    macro_clauses = [c_text for _, c_text in clause_items]
    clause_rewards: List[float] = []
    for c_name, _ in clause_items:
        rewards = clause_rewards_map.get(c_name, [])
        # Macro clause reward uses max-penalty aggregation (minimum value).
        clause_rewards.append(float(min(rewards)) if rewards else 0.0)

    return {
        "token_rewards": token_rewards,
        "node_to_token_spans": node_to_token_spans,
        "macro_clauses": macro_clauses,
        "clause_rewards": clause_rewards,
    }


def build_preorder_with_path(root: exp.Expression) -> List[NodeMeta]:
    metas: List[NodeMeta] = []

    def _dfs(node: exp.Expression, path: Tuple[str, ...]):
        clause = "ROOT" if len(path) == 0 else classify_clause(node)
        metas.append(NodeMeta(len(metas), path, node, clause))
        for key, value in node.args.items():
            if isinstance(value, exp.Expression):
                _dfs(value, path + (key,))
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, exp.Expression):
                        _dfs(item, path + (f"{key}[{idx}]",))

    _dfs(root, tuple())
    return metas


def resolve_path(root: exp.Expression, path: Tuple[str, ...]) -> Optional[exp.Expression]:
    cur: exp.Expression = root
    for seg in path:
        if "[" in seg and seg.endswith("]"):
            key = seg[: seg.index("[")]
            idx = int(seg[seg.index("[") + 1 : -1])
            arr = cur.args.get(key)
            if not isinstance(arr, list) or idx >= len(arr):
                return None
            nxt = arr[idx]
            if not isinstance(nxt, exp.Expression):
                return None
            cur = nxt
        else:
            nxt = cur.args.get(seg)
            if not isinstance(nxt, exp.Expression):
                return None
            cur = nxt
    return cur


def replace_at_path(pred_root: exp.Expression, path: Tuple[str, ...], replacement: exp.Expression) -> Optional[exp.Expression]:
    cloned = copy.deepcopy(pred_root)
    target = resolve_path(cloned, path)
    if target is None:
        return None
    target.replace(copy.deepcopy(replacement))
    return cloned


def iter_replacement_pairs(pred_root: exp.Expression, gt_root: exp.Expression) -> Iterable[Tuple[NodeMeta, exp.Expression]]:
    pred_nodes = build_preorder_with_path(pred_root)
    gt_nodes = build_preorder_with_path(gt_root)

    gt_by_kind: Dict[str, List[NodeMeta]] = {}
    for m in gt_nodes:
        gt_by_kind.setdefault(node_kind(m.node), []).append(m)

    for pred_meta in pred_nodes:
        # Skip root-level replacement; we want localized causal edits.
        if pred_meta.preorder_idx == 0:
            continue
        kind = node_kind(pred_meta.node)
        candidates = gt_by_kind.get(kind, [])
        if not candidates:
            continue

        candidates = sorted(
            candidates,
            key=lambda c: (
                0 if c.clause == pred_meta.clause else 1,
                abs(c.preorder_idx - pred_meta.preorder_idx),
            ),
        )
        best = candidates[0]
        if pred_meta.node.sql(dialect="sqlite") == best.node.sql(dialect="sqlite"):
            continue
        yield pred_meta, best.node


def fix_failure_with_earliest_repair(
    db_path: str,
    pred_sql: str,
    gt_sql: str,
    timeout_seconds: int,
) -> Optional[Dict]:
    pred_root = parse_sql(pred_sql)
    gt_root = parse_sql(gt_sql)
    if pred_root is None or gt_root is None:
        return None

    for pred_meta, gt_node in iter_replacement_pairs(pred_root, gt_root):
        modified_root = replace_at_path(pred_root, pred_meta.path, gt_node)
        if modified_root is None:
            continue
        try:
            modified_sql = modified_root.sql(dialect="sqlite")
        except Exception:
            # Some malformed AST variants can fail during dialect generation.
            # Skip this candidate and continue searching for a valid repair.
            continue
        ok, _, _ = result_equivalent(db_path, modified_sql, gt_sql, timeout_seconds)
        if ok:
            return {
                "modified_sql": modified_sql,
                "error_step": pred_meta.preorder_idx,
                "error_type": pred_meta.clause,
            }

    return None


def collect_leaf_like_nodes(root: exp.Expression) -> List[NodeMeta]:
    metas = build_preorder_with_path(root)
    candidates: List[NodeMeta] = []
    for m in metas:
        if m.preorder_idx == 0:
            continue
        if isinstance(m.node, (exp.Identifier, exp.Literal, exp.Column, exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            candidates.append(m)
    return candidates


def inject_fault_from_correct_sql(
    db_path: str,
    sql: str,
    gt_sql: str,
    rng: random.Random,
    timeout_seconds: int,
    max_trials: int = 40,
) -> Optional[Dict]:
    root = parse_sql(sql)
    if root is None:
        return None

    metas = collect_leaf_like_nodes(root)
    if not metas:
        return None

    kind_groups: Dict[str, List[NodeMeta]] = {}
    for m in metas:
        kind_groups.setdefault(node_kind(m.node), []).append(m)

    trial_pool: List[Tuple[NodeMeta, NodeMeta]] = []
    for kind, items in kind_groups.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                if items[i].path == items[j].path:
                    continue
                # Keep perturbation local by default: prefer replacing with nodes in same clause.
                if items[i].clause != items[j].clause:
                    continue
                trial_pool.append((items[i], items[j]))

    # Fallback: if strict same-clause pairing is empty, relax to same-kind only.
    if not trial_pool:
        for kind, items in kind_groups.items():
            if len(items) < 2:
                continue
            for i in range(len(items)):
                for j in range(len(items)):
                    if i == j:
                        continue
                    if items[i].path == items[j].path:
                        continue
                    trial_pool.append((items[i], items[j]))

    if not trial_pool:
        return None

    rng.shuffle(trial_pool)

    for src_meta, donor_meta in trial_pool[:max_trials]:
        modified_root = replace_at_path(root, src_meta.path, donor_meta.node)
        if modified_root is None:
            continue
        modified_sql = modified_root.sql(dialect="sqlite")
        if modified_sql.strip() == sql.strip():
            continue
        eq, _, _ = result_equivalent(db_path, modified_sql, gt_sql, timeout_seconds)
        if not eq:
            return {
                "modified_sql": modified_sql,
                "error_step": src_meta.preorder_idx,
                "error_type": src_meta.clause,
            }

    return None


def load_eval_samples(path: str) -> List[EvalSample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[EvalSample] = []
    for i, row in enumerate(data):
        question = row.get("question", "")
        db_id = row.get("db_id", "")
        gt_sql = normalize_sql_text(row.get("SQL", ""))
        pred_list = row.get("pred_sqls", [])
        pred_sql = ""
        if isinstance(pred_list, list) and pred_list:
            pred_sql = robust_extract_sql(str(pred_list[0]))

        # Keep rows with empty prediction so we can inject faults from GT directly.
        if not question or not db_id or not gt_sql:
            continue

        samples.append(EvalSample(i, db_id, question, gt_sql, pred_sql))

    return samples


def append_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("question_id")
                mode = obj.get("mode")
                if qid is not None and mode:
                    done.add((qid, mode))
            except json.JSONDecodeError:
                continue
    return done


def build_dataset(
    eval_json_path: str,
    db_root_path: str,
    output_jsonl_path: str,
    seed: int,
    timeout_seconds: int,
    max_samples: int,
    alpha: float,
    sigma: float,
):
    rng = random.Random(seed)
    samples = load_eval_samples(eval_json_path)
    if max_samples > 0:
        samples = samples[:max_samples]

    stats = BuildStats(total=len(samples))
    done = load_done_ids(output_jsonl_path)

    for s in tqdm(samples, desc="Building dataset"):
        db_path = db_path_for(db_root_path, s.db_id)
        if not os.path.exists(db_path):
            continue

        if not s.pred_sql:
            stats.pred_empty += 1
            key = (s.question_id, "inject_error")
            if key in done:
                continue
            injected = inject_fault_from_correct_sql(
                db_path=db_path,
                sql=s.gt_sql,
                gt_sql=s.gt_sql,
                rng=rng,
                timeout_seconds=timeout_seconds,
            )
            if injected:
                stats.inject_success += 1
                stats.inject_from_gt_success += 1
                reward_ann = build_causal_reward_annotations(
                    # Use faulty SQL text for reward labeling in inject mode.
                    sql=injected["modified_sql"],
                    t_star_idx=int(injected["error_step"]),
                    alpha=alpha,
                    sigma=sigma,
                )
                row = {
                    "question_id": s.question_id,
                    "db_id": s.db_id,
                    "mode": "inject_error",
                    "prompt": s.prompt,
                    "original_sql": s.gt_sql,
                    "modified_sql": injected["modified_sql"],
                    "error_step": injected["error_step"],
                    "error_type": injected["error_type"],
                    "node_to_token_spans": reward_ann["node_to_token_spans"],
                    "macro_clauses": reward_ann["macro_clauses"],
                    "clause_rewards": reward_ann["clause_rewards"],
                    "token_rewards": reward_ann["token_rewards"],
                }
                append_jsonl(output_jsonl_path, [row])
            continue

        pred_ok, _, _ = result_equivalent(db_path, s.pred_sql, s.gt_sql, timeout_seconds)

        if pred_ok:
            stats.pred_correct += 1
            key = (s.question_id, "inject_error")
            if key in done:
                continue
            injected = inject_fault_from_correct_sql(
                db_path=db_path,
                sql=s.pred_sql,
                gt_sql=s.gt_sql,
                rng=rng,
                timeout_seconds=timeout_seconds,
            )
            if injected:
                stats.inject_success += 1
                reward_ann = build_causal_reward_annotations(
                    # Use faulty SQL text for reward labeling in inject mode.
                    sql=injected["modified_sql"],
                    t_star_idx=int(injected["error_step"]),
                    alpha=alpha,
                    sigma=sigma,
                )
                row = {
                    "question_id": s.question_id,
                    "db_id": s.db_id,
                    "mode": "inject_error",
                    "prompt": s.prompt,
                    "original_sql": s.pred_sql,
                    "modified_sql": injected["modified_sql"],
                    "error_step": injected["error_step"],
                    "error_type": injected["error_type"],
                    "node_to_token_spans": reward_ann["node_to_token_spans"],
                    "macro_clauses": reward_ann["macro_clauses"],
                    "clause_rewards": reward_ann["clause_rewards"],
                    "token_rewards": reward_ann["token_rewards"],
                }
                append_jsonl(output_jsonl_path, [row])
        else:
            stats.pred_wrong += 1
            key = (s.question_id, "fix_error")
            if key in done:
                continue
            fixed = fix_failure_with_earliest_repair(
                db_path=db_path,
                pred_sql=s.pred_sql,
                gt_sql=s.gt_sql,
                timeout_seconds=timeout_seconds,
            )
            if fixed:
                stats.fix_success += 1
                reward_ann = build_causal_reward_annotations(
                    sql=s.pred_sql,
                    t_star_idx=int(fixed["error_step"]),
                    alpha=alpha,
                    sigma=sigma,
                )
                row = {
                    "question_id": s.question_id,
                    "db_id": s.db_id,
                    "mode": "fix_error",
                    "prompt": s.prompt,
                    "original_sql": s.pred_sql,
                    "modified_sql": fixed["modified_sql"],
                    "error_step": fixed["error_step"],
                    "error_type": fixed["error_type"],
                    "node_to_token_spans": reward_ann["node_to_token_spans"],
                    "macro_clauses": reward_ann["macro_clauses"],
                    "clause_rewards": reward_ann["clause_rewards"],
                    "token_rewards": reward_ann["token_rewards"],
                }
                append_jsonl(output_jsonl_path, [row])

    print("=" * 60)
    print("AST counterfactual dataset build finished")
    print(f"Total samples: {stats.total}")
    print(f"Pred correct: {stats.pred_correct}")
    print(f"Pred wrong: {stats.pred_wrong}")
    print(f"Pred empty: {stats.pred_empty}")
    print(f"Fix success: {stats.fix_success}")
    print(f"Inject success: {stats.inject_success}")
    print(f"Inject-from-GT success: {stats.inject_from_gt_success}")
    print(f"Output: {output_jsonl_path}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Build AST counterfactual dataset from eval results")
    parser.add_argument(
        "--eval_json",
        type=str,
        default="/home/banlujie/RL-NL2SQL/eval_res/eval_results_greedy_Qwen3.5-9B.json",
        help="Path to eval result json",
    )
    parser.add_argument(
        "--db_root",
        type=str,
        default="/data/banlujie/data/bird_dev/dev_databases",
        help="Root path of sqlite db folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/banlujie/data/bird_dev/bird_dev_counterfactual.jsonl",
        help="Output jsonl path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timeout_seconds", type=int, default=30, help="SQL execution timeout")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--alpha", type=float, default=1.0, help="Causal penalty alpha")
    parser.add_argument("--sigma", type=float, default=2.0, help="Causal penalty sigma")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        eval_json_path=args.eval_json,
        db_root_path=args.db_root,
        output_jsonl_path=args.output,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
        max_samples=args.max_samples,
        alpha=args.alpha,
        sigma=args.sigma,
    )
