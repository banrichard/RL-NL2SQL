import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from string import Formatter
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
if DATASET_DIR not in sys.path:
    sys.path.insert(0, DATASET_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.build_ast_counterfactual_dataset import (
    db_path_for,
    fix_failure_with_earliest_repair,
    normalize_sql_text,
    result_equivalent,
    robust_extract_sql,
)
from dataset.expand_prm_dataset import CLAUSE_ORDER, clause_expression_map, generate_multiple_mutations, parse_sql
from dataset.utils import serialize_schema


class SafeFormatDict(dict):
    def __missing__(self, key):
        return ""


def _squash_ws(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _extract_sql_field(row: Dict, keys: List[str], is_prediction: bool = False) -> str:
    for key in keys:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return normalize_sql_text(val)

    if is_prediction:
        pred_list = row.get("pred_sqls")
        if isinstance(pred_list, list) and pred_list:
            return robust_extract_sql(str(pred_list[0]))

    return ""


def _extract_schema_context(row: Dict) -> str:
    for key in ["schema", "db_details", "schema_context", "database_schema"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            return json.dumps(val, ensure_ascii=False, indent=2)
    return ""


def _extract_evidence(row: Dict) -> str:
    for key in ["evidence", "Evidence", "hint"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list):
            parts = [str(x).strip() for x in val if str(x).strip()]
            if parts:
                return " ".join(parts)
    return ""


def load_schema_map(tables_path: str) -> Dict[str, Dict]:
    if not tables_path or not os.path.exists(tables_path):
        return {}
    with open(tables_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return {}
    return {str(x.get("db_id", "")).strip(): x for x in data if isinstance(x, dict) and x.get("db_id")}


def load_template_text(template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def render_sft_template(row: Dict, template: str, schema_map: Dict[str, Dict]) -> str:
    db_id = str(row.get("db_id", "")).strip()
    db_details = ""
    if db_id and db_id in schema_map:
        try:
            db_details = serialize_schema(schema_map[db_id])
        except Exception:
            db_details = ""
    if not db_details:
        db_details = _extract_schema_context(row)

    evidence = _extract_evidence(row)
    values = SafeFormatDict(
        {
            "db_engine": row.get("db_engine", "sqlite"),
            "db_id": db_id,
            "db_details": db_details,
            "question": row.get("question", ""),
            "evidence": evidence,
        }
    )

    # Ensure unknown placeholders in templates do not crash formatting.
    formatter = Formatter()
    fields = [name for _, name, _, _ in formatter.parse(template) if name]
    for field in fields:
        if field not in values:
            values[field] = row.get(field, "")

    return template.format_map(values).strip()


def infer_target_clause_from_chosen(chosen_clause: str, base_sql: str) -> str:
    root = parse_sql(base_sql)
    if root is None:
        return "SELECT"

    clause_map = clause_expression_map(root)
    norm_chosen = _squash_ws(chosen_clause)

    for name in CLAUSE_ORDER:
        ctext = clause_map.get(name, "")
        if ctext and _squash_ws(ctext) == norm_chosen:
            return name

    up = chosen_clause.strip().upper()
    if up.startswith("SELECT"):
        return "SELECT"
    if up.startswith("FROM"):
        return "FROM"
    if up.startswith("WHERE"):
        return "WHERE"
    if up.startswith("GROUP BY"):
        return "GROUP_BY"
    if up.startswith("HAVING"):
        return "HAVING"
    if up.startswith("ORDER BY"):
        return "ORDER_BY"
    if up.startswith("LIMIT"):
        return "LIMIT"
    if up.startswith("JOIN"):
        return "JOIN"

    return "SELECT"


def extract_macro_clauses_and_prefix(sql_chosen: str, sql_rejected: str, target_clause: str) -> Dict[str, str]:
    chosen_root = parse_sql(sql_chosen)
    rejected_root = parse_sql(sql_rejected)

    if chosen_root is None or rejected_root is None:
        return {
            "macro_clause_chosen": sql_chosen,
            "macro_clause_rejected": sql_rejected,
            "prefix_context": "",
        }

    chosen_map = clause_expression_map(chosen_root)
    rejected_map = clause_expression_map(rejected_root)

    macro_clause_chosen = chosen_map.get(target_clause, sql_chosen)
    macro_clause_rejected = rejected_map.get(target_clause, sql_rejected)

    prefix_parts: List[str] = []
    for c in CLAUSE_ORDER:
        if c == target_clause:
            break
        if c in chosen_map:
            prefix_parts.append(chosen_map[c])

    return {
        "macro_clause_chosen": macro_clause_chosen,
        "macro_clause_rejected": macro_clause_rejected,
        "prefix_context": "\n".join(prefix_parts).strip(),
    }


def extract_prefix_context(sql: str, target_clause: str) -> str:
    root = parse_sql(sql)
    if root is None:
        return ""
    clause_map = clause_expression_map(root)
    parts: List[str] = []
    for c in CLAUSE_ORDER:
        if c == target_clause:
            break
        if c in clause_map:
            parts.append(clause_map[c])
    return "\n".join(parts).strip()


def build_prm_prompt(rendered_sft_prompt: str, prefix_context: str) -> str:
    return (
        f"{rendered_sft_prompt}\n"
        f"Generated SQL Prefix Context:\n{prefix_context}\n"
        f"The next SQL clause to be generated is shown below. Please evaluate its logical correctness.\n"
    ).strip()


def load_json_or_jsonl(path: str) -> List[Dict]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return [x for x in data if isinstance(x, dict)]

    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
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


def load_checkpoint(checkpoint_path: str) -> Dict:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_checkpoint(checkpoint_path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, checkpoint_path)


def _is_correct(row: Dict) -> bool:
    val = row.get("is_correct")
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y"}
    return False


def process_row_to_records(
    row: Dict,
    template: str,
    schema_map: Dict[str, Dict],
    db_root: str,
    timeout_seconds: int,
    max_mutations: int,
) -> List[Dict]:
    """Process one input record, yielding training pairs from *two* strategies.

    Strategy 2 – Multi-Path Error Harvesting:
        For every unique incorrect prediction in ``row['pred_sqls']``, run
        Graph Surgery to locate t* and emit a (chosen, rejected) pair.

    Strategy 3 – Combinatorial Fault Injection:
        Run the *unlocked* ``generate_multiple_mutations`` (all operators,
        rejection-sampled via DB execution) on the GT SQL and emit one
        training record per valid mutation.
    """
    # --- Extract GT SQL (new field: output_seq) ---
    gt_sql = ""
    for key in ["output_seq", "gt_sql", "SQL", "sql", "query"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            gt_sql = normalize_sql_text(val)
            break
    if not gt_sql:
        return []

    db_id = str(row.get("db_id", "")).strip()
    if not db_id:
        return []
    db_path = db_path_for(db_root, db_id)
    if not os.path.exists(db_path):
        return []

    rendered_sft_prompt = render_sft_template(row, template, schema_map)
    out: List[Dict] = []

    # ================================================================
    # Strategy 2 – Multi-Path Error Harvesting
    # ================================================================
    pred_sqls_raw = row.get("pred_sqls")
    if isinstance(pred_sqls_raw, list) and pred_sqls_raw:
        unique_preds = list({
            normalize_sql_text(robust_extract_sql(str(p)))
            for p in pred_sqls_raw
            if isinstance(p, str) and p.strip()
        })

        for pred_sql in unique_preds:
            if not pred_sql:
                continue
            # Execute & compare
            try:
                is_eq, _, _ = result_equivalent(db_path, pred_sql, gt_sql, timeout_seconds)
            except Exception:
                continue
            if is_eq:
                continue  # correct prediction – skip (handled by Strategy 3)

            # Incorrect → Graph Surgery to find t*
            try:
                fixed = fix_failure_with_earliest_repair(
                    db_path=db_path,
                    pred_sql=pred_sql,
                    gt_sql=gt_sql,
                    timeout_seconds=timeout_seconds,
                )
            except Exception:
                continue
            if not fixed:
                continue

            target_clause = str(fixed.get("error_type", "SELECT"))
            modified_sql = str(fixed.get("modified_sql", "")).strip()
            if not modified_sql:
                continue

            ext = extract_macro_clauses_and_prefix(
                sql_chosen=modified_sql,
                sql_rejected=pred_sql,
                target_clause=target_clause,
            )
            prm_prompt = build_prm_prompt(rendered_sft_prompt, ext["prefix_context"])
            out.append({
                "prompt": prm_prompt,
                "chosen": ext["macro_clause_chosen"],
                "rejected": ext["macro_clause_rejected"],
            })

    # ================================================================
    # Strategy 3 – Combinatorial Fault Injection (unlocked mutations)
    # ================================================================
    schema_context = _extract_schema_context(row)
    try:
        mutations = generate_multiple_mutations(
            gt_sql=gt_sql,
            schema_context=schema_context,
            max_mutations=max_mutations,
            db_path=db_path,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        mutations = []

    for m in mutations:
        chosen_clause = m.get("chosen_clause", "")
        rejected_clause = m.get("rejected_clause", "")
        if not chosen_clause or not rejected_clause:
            continue

        target_clause = infer_target_clause_from_chosen(chosen_clause, gt_sql)
        prefix_context = extract_prefix_context(gt_sql, target_clause)
        prm_prompt = build_prm_prompt(rendered_sft_prompt, prefix_context)
        out.append({
            "prompt": prm_prompt,
            "chosen": chosen_clause,
            "rejected": rejected_clause,
        })

    return out


def _worker_fn(
    row: Dict,
    template: str,
    schema_map: Dict[str, Dict],
    db_root: str,
    timeout_seconds: int,
    max_mutations: int,
) -> List[Dict]:
    """Top-level wrapper so ProcessPoolExecutor can pickle it."""
    try:
        return process_row_to_records(
            row=row,
            template=template,
            schema_map=schema_map,
            db_root=db_root,
            timeout_seconds=timeout_seconds,
            max_mutations=max_mutations,
        )
    except Exception:
        return []


def build_v1_reward_dataset(
    input_path: str,
    output_path: str,
    template_path: str,
    tables_path: str,
    db_root: str,
    timeout_seconds: int,
    max_mutations: int,
    limit: int,
    mode: str,
    checkpoint_path: str,
    checkpoint_every: int,
    start_index_override: int,
    workers: int,
) -> None:
    rows = load_json_or_jsonl(input_path)
    if limit > 0:
        rows = rows[:limit]

    template = load_template_text(template_path)
    schema_map = load_schema_map(tables_path)

    start_index = 0
    total_written = 0
    failed_rows = 0

    if mode == "fresh":
        if os.path.exists(output_path):
            os.remove(output_path)
        if checkpoint_path and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    else:
        ckpt = load_checkpoint(checkpoint_path)
        if start_index_override > 0:
            start_index = min(start_index_override, len(rows))
            print(f"Using manual start_index override: {start_index}")
        elif mode == "resume" and ckpt:
            start_index = int(ckpt.get("next_index", 0))
            total_written = int(ckpt.get("written", 0))
            failed_rows = int(ckpt.get("failed_rows", 0))
        elif mode == "resume" and os.path.exists(output_path):
            print("Warning: output exists but checkpoint missing; resume will start from index 0.")

    if start_index >= len(rows):
        print("All rows already processed according to checkpoint.")
        return

    pending_rows = rows[start_index:]
    effective_workers = max(1, workers if workers > 0 else (os.cpu_count() or 2) - 1)

    worker = partial(
        _worker_fn,
        template=template,
        schema_map=schema_map,
        db_root=db_root,
        timeout_seconds=timeout_seconds,
        max_mutations=max_mutations,
    )

    pbar = tqdm(total=len(pending_rows), desc="Building v1 PRM dataset")

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(worker, row): idx
            for idx, row in enumerate(pending_rows, start=start_index)
        }
        for future in as_completed(futures):
            local_idx = futures[future]
            try:
                recs = future.result()
            except Exception:
                recs = []

            if recs:
                try:
                    append_jsonl(output_path, recs)
                    total_written += len(recs)
                except Exception:
                    failed_rows += 1
            pbar.update(1)

            if checkpoint_path and ((local_idx + 1) % max(1, checkpoint_every) == 0):
                save_checkpoint(
                    checkpoint_path,
                    {
                        "next_index": local_idx + 1,
                        "written": total_written,
                        "failed_rows": failed_rows,
                        "input_path": input_path,
                        "output_path": output_path,
                        "mode": mode,
                    },
                )

    pbar.close()

    if checkpoint_path:
        save_checkpoint(
            checkpoint_path,
            {
                "next_index": len(rows),
                "written": total_written,
                "failed_rows": failed_rows,
                "input_path": input_path,
                "output_path": output_path,
                "mode": mode,
            },
        )

    print("=" * 60)
    print("V1 PRM dataset build finished")
    print(f"Mode: {mode}")
    print(f"Start index: {start_index}")
    print(f"Input rows: {len(rows)}")
    print(f"Output rows (written so far): {total_written}")
    print(f"Failed rows: {failed_rows}")
    print(f"Output path: {output_path}")
    if checkpoint_path:
        print(f"Checkpoint path: {checkpoint_path}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v1 PRM dataset for HuggingFace RewardTrainer")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/banlujie/RL-NL2SQL/eval_res/eval_results_voting_Qwen3.5-9B.json",
        help="Input SFT evaluation results (.json or .jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/banlujie/data/bird_dev/v1_prm_dataset.jsonl",
        help="Output PRM dataset path",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="/data/banlujie/RL-NL2SQL/templates/sql_generation.txt",
        help="SFT template file path",
    )
    parser.add_argument(
        "--db_root",
        type=str,
        default="/data/banlujie/data/bird_dev/dev_databases",
        help="SQLite DB root path",
    )
    parser.add_argument(
        "--tables_path",
        type=str,
        default="/data/banlujie/data/bird_dev/dev_tables.json",
        help="BIRD tables json path for real DDL serialization",
    )
    parser.add_argument("--timeout_seconds", type=int, default=30, help="SQL execution timeout for fixing failures")
    parser.add_argument("--max_mutations", type=int, default=5, help="Max mutations for each correct SQL")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for debug")
    parser.add_argument(
        "--mode",
        type=str,
        default="resume",
        choices=["fresh", "resume", "expand"],
        help="fresh: recreate output; resume: continue from checkpoint; expand: append without resetting",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Checkpoint json path (default: <output>.ckpt.json)",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=20,
        help="Save checkpoint every N processed input rows",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Manual start index for resuming old runs without checkpoint",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 means os.cpu_count()-1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    template_path = args.template
    if not os.path.exists(template_path):
        alt = "/home/banlujie/RL-NL2SQL/templates/sql_generation.txt"
        if os.path.exists(alt):
            template_path = alt

    tables_path = args.tables_path
    if not os.path.exists(tables_path):
        alt_tables = "/home/banlujie/data/bird_dev/dev_tables.json"
        if os.path.exists(alt_tables):
            tables_path = alt_tables

    build_v1_reward_dataset(
        input_path=args.input,
        output_path=args.output,
        template_path=template_path,
        tables_path=tables_path,
        db_root=args.db_root,
        timeout_seconds=args.timeout_seconds,
        max_mutations=max(1, args.max_mutations),
        limit=max(0, args.limit),
        mode=args.mode,
        checkpoint_path=args.checkpoint_path or (args.output + ".ckpt.json"),
        checkpoint_every=max(1, args.checkpoint_every),
        start_index_override=max(0, args.start_index),
        workers=max(0, args.workers),
    )
