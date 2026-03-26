import argparse
import copy
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import Dict, Iterable, List, Optional, Tuple

import sqlglot
from sqlglot import exp
from tqdm import tqdm

from utils import execute_sql_with_timeout


_TIMEOUT_SAMPLE_RATE = 0.1


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


@dataclass
class NodeMeta:
    preorder_idx: int
    path: Tuple[str, ...]
    node: exp.Expression
    clause: str


@dataclass
class MutationCandidate:
    path: Tuple[str, ...]
    replacement: exp.Expression
    clause: str
    error_type: str


def normalize_sql_text(sql: str) -> str:
    if not isinstance(sql, str):
        return ""
    sql = sql.strip()
    if sql.startswith("```sql") and sql.endswith("```"):
        sql = sql[6:-3].strip()
    return sql


def parse_sql(sql: str) -> Optional[exp.Expression]:
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None


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


def _select_root(root: exp.Expression) -> Optional[exp.Expression]:
    if isinstance(root, exp.Select):
        return root
    return root.find(exp.Select)


def clause_expression_map(root: exp.Expression) -> Dict[str, str]:
    sroot = _select_root(root)
    if sroot is None:
        return {}

    clauses: Dict[str, str] = {}

    select_exprs = sroot.args.get("expressions") or []
    if select_exprs:
        select_sql = ", ".join(expr.sql(dialect="sqlite") for expr in select_exprs)
        clauses["SELECT"] = f"SELECT {select_sql}"

    from_expr = sroot.args.get("from")
    if isinstance(from_expr, exp.Expression):
        clauses["FROM"] = from_expr.sql(dialect="sqlite")

    joins = sroot.args.get("joins") or []
    if joins:
        join_texts = [j.sql(dialect="sqlite") for j in joins if isinstance(j, exp.Expression)]
        if join_texts:
            clauses["JOIN"] = " ".join(join_texts)

    where_expr = sroot.args.get("where")
    if isinstance(where_expr, exp.Expression):
        clauses["WHERE"] = where_expr.sql(dialect="sqlite")

    group_expr = sroot.args.get("group")
    if isinstance(group_expr, exp.Expression):
        clauses["GROUP_BY"] = group_expr.sql(dialect="sqlite")

    having_expr = sroot.args.get("having")
    if isinstance(having_expr, exp.Expression):
        clauses["HAVING"] = having_expr.sql(dialect="sqlite")

    order_expr = sroot.args.get("order")
    if isinstance(order_expr, exp.Expression):
        clauses["ORDER_BY"] = order_expr.sql(dialect="sqlite")

    limit_expr = sroot.args.get("limit")
    if isinstance(limit_expr, exp.Expression):
        clauses["LIMIT"] = limit_expr.sql(dialect="sqlite")

    return clauses


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


def replace_at_path(root: exp.Expression, path: Tuple[str, ...], replacement: exp.Expression) -> Optional[exp.Expression]:
    cloned = copy.deepcopy(root)
    target = resolve_path(cloned, path)
    if target is None:
        return None
    target.replace(copy.deepcopy(replacement))
    return cloned


def _flip_operator(node: exp.Expression) -> Optional[exp.Expression]:
    if isinstance(node, exp.GT):
        return exp.LTE(this=copy.deepcopy(node.this), expression=copy.deepcopy(node.expression))
    if isinstance(node, exp.LT):
        return exp.GTE(this=copy.deepcopy(node.this), expression=copy.deepcopy(node.expression))
    if isinstance(node, exp.EQ):
        return exp.NEQ(this=copy.deepcopy(node.this), expression=copy.deepcopy(node.expression))
    return None


def _swap_conjunction(node: exp.Expression) -> Optional[exp.Expression]:
    if isinstance(node, exp.And):
        return exp.Or(this=copy.deepcopy(node.this), expression=copy.deepcopy(node.expression))
    if isinstance(node, exp.Or):
        return exp.And(this=copy.deepcopy(node.this), expression=copy.deepcopy(node.expression))
    return None


def _mutate_aggregation(node: exp.Expression) -> Optional[exp.Expression]:
    if isinstance(node, exp.Count):
        return exp.Sum(this=copy.deepcopy(node.this))
    if isinstance(node, exp.Sum):
        return exp.Max(this=copy.deepcopy(node.this))
    if isinstance(node, exp.Max):
        return exp.Min(this=copy.deepcopy(node.this))
    if isinstance(node, exp.Min):
        return exp.Count(this=copy.deepcopy(node.this))
    return None


def _typo_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "col_err"
    if len(name) > 3 and name.lower().endswith("s"):
        return name[:-1]
    if name.lower().endswith("_err"):
        return name + "1"
    return name + "_err"


def _mutate_identifier(node: exp.Expression) -> Optional[exp.Expression]:
    if isinstance(node, exp.Column):
        new_col = copy.deepcopy(node)
        this = new_col.args.get("this")
        if isinstance(this, exp.Identifier):
            this.set("this", _typo_name(this.this))
            return new_col
        if isinstance(this, str):
            new_col.set("this", _typo_name(this))
            return new_col
        return None

    if isinstance(node, exp.Table):
        new_table = copy.deepcopy(node)
        this = new_table.args.get("this")
        if isinstance(this, exp.Identifier):
            this.set("this", _typo_name(this.this))
            return new_table
        if isinstance(this, str):
            new_table.set("this", _typo_name(this))
            return new_table
        return None

    return None


def _classify_execution_error(error_str: str) -> str:
    """Classify an execution error string into SCHEMA_ERROR, TIMEOUT_ERROR, or LOGIC_ERROR."""
    lower = error_str.lower()
    if "timed out" in lower or "timeout" in lower:
        return "TIMEOUT_ERROR"
    if any(kw in lower for kw in ("no such table", "no such column", "has no column")):
        return "SCHEMA_ERROR"
    return "LOGIC_ERROR"


def generate_multiple_mutations(
    gt_sql: str,
    schema_context: str,
    max_mutations: int = 5,
    db_path: Optional[str] = None,
    timeout_seconds: int = 30,
) -> List[Dict]:
    """Generate ALL valid AST mutations with optional rejection sampling.

    When *db_path* is provided the mutated SQL is executed against the DB and
    filtered through rejection-sampling guardrails:
      - LOGIC_ERROR  (wrong result) -> always keep
      - SCHEMA_ERROR (missing table / column) -> always keep
      - TIMEOUT_ERROR -> keep with 10 % probability
    Mutations that accidentally produce the correct result are discarded.

    When *db_path* is ``None`` the function falls back to the previous
    AST-only validation (parseability check, no execution).
    """
    gt_sql = normalize_sql_text(gt_sql)
    root = parse_sql(gt_sql)
    if root is None:
        return []

    metas = build_preorder_with_path(root)
    candidates: List[MutationCandidate] = []

    for m in metas:
        if m.preorder_idx == 0:
            continue

        # --- Operator Flip (WHERE / HAVING comparisons) ---
        if m.clause in {"WHERE", "HAVING"} and isinstance(m.node, (exp.GT, exp.LT, exp.EQ)):
            repl = _flip_operator(m.node)
            if repl is not None:
                candidates.append(
                    MutationCandidate(
                        path=m.path,
                        replacement=repl,
                        clause=m.clause,
                        error_type="operator_flipping",
                    )
                )

        # --- Conjunction Swap (AND <-> OR) ---
        if isinstance(m.node, (exp.And, exp.Or)):
            repl = _swap_conjunction(m.node)
            if repl is not None:
                candidates.append(
                    MutationCandidate(
                        path=m.path,
                        replacement=repl,
                        clause=m.clause,
                        error_type="conjunction_swapping",
                    )
                )

        # --- Aggregation Mutation (any clause, not just SELECT) ---
        if isinstance(m.node, (exp.Count, exp.Sum, exp.Max, exp.Min)):
            repl = _mutate_aggregation(m.node)
            if repl is not None:
                candidates.append(
                    MutationCandidate(
                        path=m.path,
                        replacement=repl,
                        clause=m.clause,
                        error_type="aggregation_mutation",
                    )
                )

        # --- Identifier Typo ---
        if isinstance(m.node, (exp.Column, exp.Table)):
            repl = _mutate_identifier(m.node)
            if repl is not None:
                candidates.append(
                    MutationCandidate(
                        path=m.path,
                        replacement=repl,
                        clause=m.clause,
                        error_type="identifier_typo",
                    )
                )

    if not candidates:
        return []

    # Pre-compute GT result once when rejection sampling is active.
    gt_result = None
    if db_path:
        gt_result = execute_sql_with_timeout(db_path, gt_sql, timeout_seconds=timeout_seconds)

    original_clause_map = clause_expression_map(root)

    results: List[Dict] = []
    seen_sql: set = set()

    for cand in candidates:
        mutated_root = replace_at_path(root, cand.path, cand.replacement)
        if mutated_root is None:
            continue

        try:
            mutated_sql = mutated_root.sql(dialect="sqlite")
        except Exception:
            continue

        if not mutated_sql or mutated_sql == gt_sql:
            continue
        if mutated_sql in seen_sql:
            continue

        # Ensure mutated SQL is still parseable as SQL text.
        if parse_sql(mutated_sql) is None:
            continue

        # ---- Rejection sampling via DB execution ----
        if db_path:
            mut_result = execute_sql_with_timeout(db_path, mutated_sql, timeout_seconds=timeout_seconds)
            is_error = isinstance(mut_result, str) and mut_result.startswith("Error")

            if is_error:
                err_class = _classify_execution_error(mut_result)
                if err_class == "TIMEOUT_ERROR" and random.random() > _TIMEOUT_SAMPLE_RATE:
                    continue  # drop 90 % of timeouts
                # SCHEMA_ERROR and LOGIC_ERROR pass through
            else:
                # Mutation executed successfully – compare results.
                if mut_result == gt_result:
                    continue  # still correct, not a useful contrastive pair

        mutated_clause_map = clause_expression_map(mutated_root)
        chosen_clause = original_clause_map.get(cand.clause, gt_sql)
        rejected_clause = mutated_clause_map.get(cand.clause, mutated_sql)

        results.append(
            {
                "chosen_clause": chosen_clause,
                "rejected_clause": rejected_clause,
                "error_type": cand.error_type,
            }
        )
        seen_sql.add(mutated_sql)

    return results


def _extract_gt_sql(row: Dict) -> str:
    for key in ["gt_sql", "SQL", "sql", "query"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return normalize_sql_text(val)
    return ""


def _extract_schema_context(row: Dict) -> str:
    for key in ["schema_context", "db_schema", "schema", "prompt"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    db_id = str(row.get("db_id", "")).strip()
    question = str(row.get("question", "")).strip()
    parts = []
    if db_id:
        parts.append(f"DB_ID: {db_id}")
    if question:
        parts.append(f"Question: {question}")
    return "\n".join(parts)


def process_row(row: Dict, max_mutations: int) -> List[Dict]:
    gt_sql = _extract_gt_sql(row)
    if not gt_sql:
        return []
    schema_context = _extract_schema_context(row)
    return generate_multiple_mutations(gt_sql, schema_context, max_mutations=max_mutations)


def load_rows(input_path: str) -> List[Dict]:
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return [x for x in data if isinstance(x, dict)]

    rows: List[Dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def append_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def expand_dataset(
    input_path: str,
    output_path: str,
    max_mutations: int = 5,
    workers: Optional[int] = None,
    chunksize: int = 16,
) -> None:
    rows = load_rows(input_path)
    total = len(rows)
    if total == 0:
        print("No valid input rows found.")
        return

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    if os.path.exists(output_path):
        os.remove(output_path)

    total_written = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        mapped = executor.map(process_row, rows, repeat(max_mutations), chunksize=chunksize)
        for mutation_list in tqdm(mapped, total=total, desc="Expanding rows"):
            if not mutation_list:
                continue
            append_jsonl(output_path, mutation_list)
            total_written += len(mutation_list)

    print("=" * 60)
    print("Combinatorial fault injection finished")
    print(f"Input rows: {total}")
    print(f"Output rows: {total_written}")
    print(f"Workers: {workers}")
    print(f"Output file: {output_path}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand PRM dataset with combinatorial AST mutations")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/banlujie/RL-NL2SQL/eval_res/eval_results_greedy_Qwen3.5-9B.json",
        help="Input JSON/JSONL with GT SQL rows",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/banlujie/RL-NL2SQL/eval_res/expanded_prm_dataset.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--max_mutations", type=int, default=5, help="Max distinct mutations per GT SQL")
    parser.add_argument("--workers", type=int, default=0, help="Worker processes (0 means os.cpu_count()-1)")
    parser.add_argument("--chunksize", type=int, default=16, help="ProcessPool map chunksize")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    workers = None if args.workers <= 0 else args.workers
    expand_dataset(
        input_path=args.input,
        output_path=args.output,
        max_mutations=max(1, args.max_mutations),
        workers=workers,
        chunksize=max(1, args.chunksize),
    )
