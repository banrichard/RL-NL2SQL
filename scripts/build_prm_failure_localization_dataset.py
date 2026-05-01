import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from build_v1_reward_dataset import (
    build_prm_prompt,
    extract_macro_clauses_and_prefix,
    extract_prefix_context,
    infer_target_clause_from_chosen,
    load_json_or_jsonl,
    load_schema_map,
    load_template_text,
    perform_graph_surgery,
    render_sft_template,
)
from dataset.build_ast_counterfactual_dataset import (
    db_path_for,
    fix_failure_with_earliest_repair,
    is_timeout_error,
    normalize_sql_text,
    parse_sql,
    result_equivalent,
    robust_extract_sql,
)
from dataset.expand_prm_dataset import CLAUSE_ORDER, clause_expression_map, generate_multiple_mutations


def _normalize_db_roots(source: Dict[str, Any], default_db_root: str) -> List[str]:
    roots: List[str] = []
    db_roots = source.get("db_roots")
    if isinstance(db_roots, list):
        roots.extend([str(x).strip() for x in db_roots if str(x).strip()])
    db_root = source.get("db_root")
    if isinstance(db_root, str) and db_root.strip():
        roots.append(db_root.strip())
    if not roots and default_db_root:
        roots.append(default_db_root)

    deduped: List[str] = []
    seen = set()
    for r in roots:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def _resolve_db_path(db_id: str, db_roots: List[str]) -> str:
    db_id = (db_id or "").strip()
    if not db_id:
        return ""
    for root in db_roots:
        root = (root or "").strip()
        if not root:
            continue
        candidates = [
            os.path.join(root, db_id, f"{db_id}.sqlite"),
            os.path.join(root, f"{db_id}.sqlite"),
            os.path.join(root, "databases", db_id, f"{db_id}.sqlite"),
            os.path.join(root, "train_databases", db_id, f"{db_id}.sqlite"),
            os.path.join(root, "dev_databases", db_id, f"{db_id}.sqlite"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
    return ""


def _load_sources(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.sources_config and os.path.exists(args.sources_config):
        with open(args.sources_config, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("--sources-config must be a JSON list.")
        sources: List[Dict[str, Any]] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            input_path = str(item.get("input", "")).strip()
            if not input_path:
                raise ValueError(f"source[{i}] missing required field: input")
            sources.append(
                {
                    "name": str(item.get("name", f"source_{i}")).strip() or f"source_{i}",
                    "input": input_path,
                    "tables_path": str(item.get("tables_path", args.tables_path)).strip(),
                    "template_path": str(item.get("template_path", args.template_path)).strip(),
                    "db_roots": _normalize_db_roots(item, args.db_root),
                    "max_samples": int(item.get("max_samples", 0) or 0),
                }
            )
        return sources

    return [
        {
            "name": "default",
            "input": args.input,
            "tables_path": args.tables_path,
            "template_path": args.template_path,
            "db_roots": [args.db_root] if args.db_root else [],
            "max_samples": 0,
        }
    ]


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _example_uid(ex: Dict[str, Any]) -> str:
    base = f"{ex.get('source_name','')}\n{ex.get('question_id','')}\n{ex.get('db_id','')}\n{ex.get('gt_sql','')}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _load_checkpoint(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_existing_summary(path: str) -> Tuple[Counter, Counter, List[Dict[str, Any]], int]:
    if not path or not os.path.exists(path):
        return Counter(), Counter(), [], 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return Counter(), Counter(), [], 0

    if not isinstance(payload, dict):
        return Counter(), Counter(), [], 0

    stats = Counter()
    for key in [
        "parse_failures",
        "prediction_empty_count",
        "natural_failure_examples_kept",
        "fault_injection_examples_kept",
        "localization_examples_kept",
    ]:
        stats[key] = int(payload.get(key, 0) or 0)

    output_rows = payload.get("output_rows", {})
    if isinstance(output_rows, dict):
        stats["output_rows_prm_train"] = int(output_rows.get("prm_train", 0) or 0)
        stats["output_rows_prm_eval"] = int(output_rows.get("prm_eval", 0) or 0)
        stats["output_rows_failure_localization_eval"] = int(
            output_rows.get("failure_localization_eval", 0) or 0
        )

    split_rows = payload.get("split_rows", {})
    if isinstance(split_rows, dict):
        for key in [
            "natural_failure_train",
            "natural_failure_eval",
            "fault_injection_train",
            "fault_injection_eval",
        ]:
            stats[key] = int(split_rows.get(key, 0) or 0)

    failure_reasons = payload.get("failure_reasons", {})
    failure_counter = Counter(failure_reasons if isinstance(failure_reasons, dict) else {})

    existing_sources = payload.get("sources", [])
    normalized_sources = existing_sources if isinstance(existing_sources, list) else []
    total_examples = int(payload.get("total_examples", 0) or 0)
    return stats, failure_counter, normalized_sources, total_examples


def _merge_summary_sources(existing_sources: List[Dict[str, Any]], new_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()

    for item in list(existing_sources) + list(new_sources):
        if not isinstance(item, dict):
            continue
        normalized = {
            "name": str(item.get("name", "")).strip(),
            "input": str(item.get("input", "")).strip(),
            "tables_path": str(item.get("tables_path", "")).strip(),
            "template_path": str(item.get("template_path", "")).strip(),
            "db_roots": [str(x).strip() for x in item.get("db_roots", []) if str(x).strip()],
        }
        key = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)

    return merged


def _infer_resume_index_from_outputs(
    output_dir: str,
    current_source_names: List[str],
) -> int:
    if not output_dir or not current_source_names:
        return 0

    source_name_set = set(current_source_names)
    max_row_index = -1
    for filename in ["v1_prm_train.jsonl", "v1_prm_eval.jsonl"]:
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                meta = row.get("metadata")
                if not isinstance(meta, dict):
                    continue
                source_name = str(meta.get("source_dataset", "")).strip()
                if source_name not in source_name_set:
                    continue
                row_index = meta.get("row_index")
                if isinstance(row_index, int):
                    max_row_index = max(max_row_index, row_index)
    return max_row_index + 1 if max_row_index >= 0 else 0


def _extract_gt_sql(row: Dict[str, Any]) -> str:
    for key in ["output_seq", "gt_sql", "SQL", "sql", "query"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return normalize_sql_text(val)
    return ""


def _extract_schema_context(row: Dict[str, Any]) -> str:
    for key in ["schema", "db_details", "schema_context", "database_schema"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            return json.dumps(val, ensure_ascii=False)
    return ""


def _extract_pred_sqls(row: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    pred_sqls = row.get("pred_sqls")
    if isinstance(pred_sqls, list):
        for item in pred_sqls:
            if isinstance(item, str) and item.strip():
                sql = normalize_sql_text(robust_extract_sql(item))
                if sql:
                    out.append(sql)

    for key in ["pred_sql", "prediction", "response"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            sql = normalize_sql_text(robust_extract_sql(val))
            if sql:
                out.append(sql)

    seen = set()
    deduped = []
    for sql in out:
        if sql not in seen:
            seen.add(sql)
            deduped.append(sql)
    return deduped


def _stable_question_id(row: Dict[str, Any], gt_sql: str) -> Tuple[Any, str, bool]:
    qid = row.get("question_id")
    if qid is not None and str(qid).strip():
        qid_text = str(qid).strip()
        return qid, f"question_id::{qid_text}", False

    db_id = str(row.get("db_id", "")).strip()
    question = str(row.get("question", "")).strip()
    base = f"{db_id}\n{question}\n{gt_sql}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    derived = f"derived_{digest}"
    return derived, f"derived::{digest}", True


def _split_keys(example_keys: List[str], train_ratio: float, seed: int) -> set:
    import random

    keys = sorted(set(example_keys))
    rng = random.Random(seed)
    rng.shuffle(keys)

    train_count = int(len(keys) * train_ratio)
    if keys and train_ratio > 0 and train_count == 0:
        train_count = 1
    if train_count > len(keys):
        train_count = len(keys)
    return set(keys[:train_count])


def _ordered_clauses_and_prefixes(sql: str) -> Tuple[List[str], List[str], List[str]]:
    root = parse_sql(sql)
    if root is None:
        return [], [], []
    cmap = clause_expression_map(root)
    clause_names: List[str] = []
    clauses: List[str] = []
    prefixes: List[str] = []
    for name in CLAUSE_ORDER:
        ctext = cmap.get(name)
        if not ctext:
            continue
        clause_names.append(name)
        clauses.append(ctext)
        prefixes.append(extract_prefix_context(sql, name))
    return clause_names, clauses, prefixes


def _remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def build_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    sources = _load_sources(args)
    current_source_names = [str(source["name"]).strip() for source in sources]
    source_runtime: Dict[str, Dict[str, Any]] = {}
    combined_rows: List[Tuple[Dict[str, Any], str]] = []

    for source in sources:
        source_name = source["name"]
        rows = load_json_or_jsonl(source["input"])
        source_limit = int(source.get("max_samples", 0))
        if source_limit > 0:
            rows = rows[:source_limit]
        if args.smoke:
            rows = rows[: min(len(rows), 30)]
        combined_rows.extend([(r, source_name) for r in rows])
        source_runtime[source_name] = {
            "tables_path": source["tables_path"],
            "template_path": source["template_path"],
            "db_roots": source["db_roots"],
            "schema_map": load_schema_map(source["tables_path"]) if source["tables_path"] else {},
            "template": load_template_text(source["template_path"]),
            "input": source["input"],
        }

    if args.max_samples > 0:
        combined_rows = combined_rows[: args.max_samples]

    prepared: List[Dict[str, Any]] = []
    stats = Counter()
    failure_reasons = Counter()
    stats["source_count"] = len(sources)

    for idx, (row, source_name) in enumerate(combined_rows):
        db_id = str(row.get("db_id", "")).strip()
        gt_sql = _extract_gt_sql(row)
        if not gt_sql:
            failure_reasons[f"{source_name}:missing_gt_sql"] += 1
            continue
        if not db_id:
            failure_reasons[f"{source_name}:missing_db_id"] += 1
            continue

        question_id, split_key, is_derived_qid = _stable_question_id(row, gt_sql)
        prepared.append(
            {
                "row_index": idx,
                "row": row,
                "question_id": question_id,
                "split_key": split_key,
                "db_id": db_id,
                "gt_sql": gt_sql,
                "question": str(row.get("question", "")),
                "is_derived_qid": is_derived_qid,
                "source_name": source_name,
            }
        )
        prepared[-1]["split_key"] = f"{source_name}::{split_key}"
        stats[f"input_rows_{source_name}"] += 1
        if is_derived_qid:
            stats["derived_question_id_examples"] += 1

    train_keys = _split_keys([x["split_key"] for x in prepared], args.train_ratio, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    prm_train_path = os.path.join(args.output_dir, "v1_prm_train.jsonl")
    prm_eval_path = os.path.join(args.output_dir, "v1_prm_eval.jsonl")
    localization_path = os.path.join(args.output_dir, "failure_localization_eval.jsonl")
    summary_path = os.path.join(args.output_dir, "build_summary.json")
    checkpoint_path = args.checkpoint_path.strip() if args.checkpoint_path else os.path.join(
        args.output_dir, "build_prm_failure_localization.ckpt.json"
    )
    start_index = 0
    existing_summary_sources: List[Dict[str, Any]] = []
    existing_total_examples = 0

    if args.append:
        existing_stats, existing_failure_reasons, existing_summary_sources, existing_total_examples = (
            _load_existing_summary(summary_path)
        )
        if existing_stats or existing_failure_reasons or existing_summary_sources or existing_total_examples:
            stats.update(existing_stats)
            failure_reasons.update(existing_failure_reasons)
            print(f"[append] loaded existing summary: {summary_path} (total_examples={existing_total_examples})")
        else:
            print(f"[append] no existing summary found, appending into a fresh output dir: {args.output_dir}")

    if args.resume:
        ckpt = _load_checkpoint(checkpoint_path)
        if ckpt:
            start_index = min(int(ckpt.get("next_index", 0)), len(prepared))
            stats = Counter(ckpt.get("stats", {}))
            failure_reasons = Counter(ckpt.get("failure_reasons", {}))
            print(f"[resume] loaded checkpoint: {checkpoint_path} (next_index={start_index})")
        else:
            if args.append:
                raise FileNotFoundError(f"--resume requested but checkpoint not found: {checkpoint_path}")
            print(f"[resume] checkpoint not found, start from scratch: {checkpoint_path}")
        inferred_start_index = min(_infer_resume_index_from_outputs(args.output_dir, current_source_names), len(prepared))
        if inferred_start_index > start_index:
            print(
                f"[resume] advanced start_index from {start_index} to {inferred_start_index} "
                f"based on existing output rows in {args.output_dir}"
            )
            start_index = inferred_start_index
    else:
        if not args.append:
            for p in [prm_train_path, prm_eval_path, localization_path, summary_path, checkpoint_path]:
                _remove_if_exists(p)

    pbar = tqdm(
        prepared[start_index:],
        total=len(prepared),
        initial=start_index,
        desc="Building PRM + localization datasets",
    )

    for offset, ex in enumerate(pbar, start=start_index):
        uid = _example_uid(ex)
        stats["processed_examples"] += 1
        row = ex["row"]
        db_id = ex["db_id"]
        gt_sql = ex["gt_sql"]
        question_id = ex["question_id"]
        source_name = ex["source_name"]
        runtime = source_runtime[source_name]
        split = "train" if ex["split_key"] in train_keys else "eval"
        db_path = _resolve_db_path(db_id, runtime["db_roots"])

        rendered_prompt = render_sft_template(row, runtime["template"], runtime["schema_map"])
        if not rendered_prompt and isinstance(row.get("prompt"), str) and row.get("prompt", "").strip():
            rendered_prompt = row["prompt"].strip()
        question = ex["question"]
        rows_to_train: List[Dict[str, Any]] = []
        rows_to_eval: List[Dict[str, Any]] = []
        rows_to_localize: List[Dict[str, Any]] = []

        pred_sqls = _extract_pred_sqls(row)
        if not pred_sqls:
            stats["prediction_empty_count"] += 1
            failure_reasons[f"{source_name}:prediction_empty"] += 1

        # Natural failures from model predictions.
        for pred_sql in pred_sqls:
            if not db_path:
                failure_reasons[f"{source_name}:db_not_found_for_natural"] += 1
                continue
            is_eq, eq_reason, eq_info = result_equivalent(
                db_path,
                pred_sql,
                gt_sql,
                timeout_seconds=args.sql_timeout,
            )
            if is_eq:
                continue

            chosen_clause = ""
            rejected_clause = ""
            target_clause = ""
            prefix_context = ""
            alignment_strategy = ""
            skip_earliest_repair = args.skip_timeout_repair and is_timeout_error(eq_reason, eq_info)

            fixed = None
            if skip_earliest_repair:
                stats["timeout_repair_skipped"] += 1
                failure_reasons[f"{source_name}:timeout_repair_skipped"] += 1
            else:
                fixed = fix_failure_with_earliest_repair(
                    db_path=db_path,
                    pred_sql=pred_sql,
                    gt_sql=gt_sql,
                    timeout_seconds=args.sql_timeout,
                )
            if fixed:
                target_clause = str(fixed.get("error_type", "SELECT"))
                modified_sql = str(fixed.get("modified_sql", "")).strip()
                if not modified_sql:
                    failure_reasons[f"{source_name}:earliest_repair_empty_modified_sql"] += 1
                    continue
                ext = extract_macro_clauses_and_prefix(
                    sql_chosen=modified_sql,
                    sql_rejected=pred_sql,
                    target_clause=target_clause,
                )
                chosen_clause = ext["macro_clause_chosen"]
                rejected_clause = ext["macro_clause_rejected"]
                prefix_context = ext["prefix_context"]
                alignment_strategy = "earliest_repair"
            else:
                surgery = perform_graph_surgery(pred_sql, gt_sql)
                if surgery is None:
                    failure_reasons[f"{source_name}:natural_alignment_failed"] += 1
                    continue
                target_clause = surgery["divergent_clause"]
                chosen_clause = surgery["chosen_clause"]
                rejected_clause = surgery["rejected_clause"]
                prefix_context = surgery["prefix_context"]
                alignment_strategy = "graph_surgery"

            prm_prompt = build_prm_prompt(rendered_prompt, prefix_context)
            prm_row = {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "prompt": prm_prompt,
                "prefix_context": prefix_context,
                "chosen": chosen_clause,
                "rejected": rejected_clause,
                "chosen_clause": chosen_clause,
                "rejected_clause": rejected_clause,
                "target_clause": target_clause,
                "gt_sql": gt_sql,
                "pred_sql": pred_sql,
                "source_type": "natural_failure",
                "split": split,
                "metadata": {
                    "row_index": ex["row_index"],
                    "alignment_strategy": alignment_strategy,
                    "equivalence_reason": eq_reason,
                    "equivalence_info": eq_info,
                    "question_id_is_derived": ex["is_derived_qid"],
                    "source_dataset": source_name,
                    "example_uid": uid,
                },
            }
            if split == "train":
                rows_to_train.append(prm_row)
            else:
                rows_to_eval.append(prm_row)
            stats["natural_failure_examples_kept"] += 1
            stats[f"natural_failure_{split}"] += 1

            if split == "eval":
                clause_names, clauses, prefixes = _ordered_clauses_and_prefixes(pred_sql)
                if not clauses:
                    stats["parse_failures"] += 1
                    failure_reasons[f"{source_name}:localization_parse_failed_natural"] += 1
                else:
                    root_idx = clause_names.index(target_clause) if target_clause in clause_names else -1
                    if root_idx < 0:
                        failure_reasons[f"{source_name}:localization_root_not_in_candidates_natural"] += 1
                    rows_to_localize.append(
                        {
                            "question_id": question_id,
                            "db_id": db_id,
                            "question": question,
                            "prompt": rendered_prompt,
                            "gt_sql": gt_sql,
                            "pred_sql": pred_sql,
                            "clauses": clauses,
                            "prefix_contexts": prefixes,
                            "root_cause_clause_index": root_idx,
                            "root_cause_clause_name": target_clause,
                            "source_type": "natural_failure",
                            "metadata": {
                                "alignment_strategy": alignment_strategy,
                                "candidate_clause_names": clause_names,
                                "equivalence_reason": eq_reason,
                                "source_dataset": source_name,
                                "example_uid": uid,
                            },
                        }
                    )
                    stats["localization_examples_kept"] += 1

        # Synthetic failures from AST fault injection on GT SQL.
        mutations = generate_multiple_mutations(
            gt_sql=gt_sql,
            schema_context=_extract_schema_context(row),
            max_mutations=5,
            db_path=db_path if db_path else None,
            timeout_seconds=args.sql_timeout,
        )
        for m in mutations:
            chosen_clause = str(m.get("chosen_clause", "")).strip()
            rejected_clause = str(m.get("rejected_clause", "")).strip()
            pred_sql = str(m.get("mutated_sql", "")).strip()
            if not chosen_clause or not rejected_clause:
                failure_reasons[f"{source_name}:fault_injection_missing_clause_pair"] += 1
                continue
            if not pred_sql:
                failure_reasons[f"{source_name}:fault_injection_missing_mutated_sql"] += 1
                continue

            target_clause = str(m.get("target_clause") or infer_target_clause_from_chosen(chosen_clause, gt_sql))
            prefix_context = extract_prefix_context(gt_sql, target_clause)
            prm_prompt = build_prm_prompt(rendered_prompt, prefix_context)

            prm_row = {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "prompt": prm_prompt,
                "prefix_context": prefix_context,
                "chosen": chosen_clause,
                "rejected": rejected_clause,
                "chosen_clause": chosen_clause,
                "rejected_clause": rejected_clause,
                "target_clause": target_clause,
                "gt_sql": gt_sql,
                "pred_sql": pred_sql,
                "source_type": "fault_injection",
                "split": split,
                "metadata": {
                    "row_index": ex["row_index"],
                    "mutation_error_type": m.get("error_type"),
                    "question_id_is_derived": ex["is_derived_qid"],
                    "source_dataset": source_name,
                    "example_uid": uid,
                },
            }
            if split == "train":
                rows_to_train.append(prm_row)
            else:
                rows_to_eval.append(prm_row)
            stats["fault_injection_examples_kept"] += 1
            stats[f"fault_injection_{split}"] += 1

            if split == "eval":
                clause_names, clauses, prefixes = _ordered_clauses_and_prefixes(pred_sql)
                if not clauses:
                    stats["parse_failures"] += 1
                    failure_reasons[f"{source_name}:localization_parse_failed_fault_injection"] += 1
                else:
                    root_idx = clause_names.index(target_clause) if target_clause in clause_names else -1
                    if root_idx < 0:
                        failure_reasons[f"{source_name}:localization_root_not_in_candidates_fault_injection"] += 1
                    rows_to_localize.append(
                        {
                            "question_id": question_id,
                            "db_id": db_id,
                            "question": question,
                            "prompt": rendered_prompt,
                            "gt_sql": gt_sql,
                            "pred_sql": pred_sql,
                            "clauses": clauses,
                            "prefix_contexts": prefixes,
                            "root_cause_clause_index": root_idx,
                            "root_cause_clause_name": target_clause,
                            "source_type": "fault_injection",
                            "metadata": {
                                "mutation_error_type": m.get("error_type"),
                                "candidate_clause_names": clause_names,
                                "source_dataset": source_name,
                                "example_uid": uid,
                            },
                        }
                    )
                    stats["localization_examples_kept"] += 1

        if rows_to_train:
            append_jsonl(prm_train_path, rows_to_train)
            stats["output_rows_prm_train"] += len(rows_to_train)
        if rows_to_eval:
            append_jsonl(prm_eval_path, rows_to_eval)
            stats["output_rows_prm_eval"] += len(rows_to_eval)
        if rows_to_localize:
            append_jsonl(localization_path, rows_to_localize)
            stats["output_rows_failure_localization_eval"] += len(rows_to_localize)

        if (offset + 1) % max(1, args.checkpoint_every) == 0:
            _save_checkpoint(
                checkpoint_path,
                {
                    "next_index": offset + 1,
                    "stats": dict(stats),
                    "failure_reasons": dict(failure_reasons),
                    "args": {
                        "sources_config": args.sources_config,
                        "input": args.input,
                        "output_dir": args.output_dir,
                        "train_ratio": args.train_ratio,
                        "seed": args.seed,
                        "max_samples": args.max_samples,
                        "smoke": args.smoke,
                    },
                },
            )

    _save_checkpoint(
        checkpoint_path,
        {
            "next_index": len(prepared),
            "stats": dict(stats),
            "failure_reasons": dict(failure_reasons),
            "completed": True,
        },
    )

    current_sources = [
        {
            "name": s["name"],
            "input": s["input"],
            "tables_path": s["tables_path"],
            "template_path": s["template_path"],
            "db_roots": s["db_roots"],
        }
        for s in sources
    ]

    summary = {
        "input_path": args.input,
        "tables_path": args.tables_path,
        "db_root": args.db_root,
        "template_path": args.template_path,
        "sources": _merge_summary_sources(existing_summary_sources, current_sources),
        "output_dir": args.output_dir,
        "total_examples": existing_total_examples + len(prepared),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "smoke": args.smoke,
        "resume": bool(args.resume),
        "append": bool(args.append),
        "checkpoint_path": checkpoint_path,
        "checkpoint_every": args.checkpoint_every,
        "sql_timeout": args.sql_timeout,
        "skip_timeout_repair": bool(args.skip_timeout_repair),
        "start_index": start_index,
        "parse_failures": int(stats["parse_failures"]),
        "prediction_empty_count": int(stats["prediction_empty_count"]),
        "natural_failure_examples_kept": int(stats["natural_failure_examples_kept"]),
        "fault_injection_examples_kept": int(stats["fault_injection_examples_kept"]),
        "localization_examples_kept": int(stats["localization_examples_kept"]),
        "output_rows": {
            "prm_train": int(stats["output_rows_prm_train"]),
            "prm_eval": int(stats["output_rows_prm_eval"]),
            "failure_localization_eval": int(stats["output_rows_failure_localization_eval"]),
        },
        "split_rows": {
            "natural_failure_train": int(stats["natural_failure_train"]),
            "natural_failure_eval": int(stats["natural_failure_eval"]),
            "fault_injection_train": int(stats["fault_injection_train"]),
            "fault_injection_eval": int(stats["fault_injection_eval"]),
        },
        "failure_reasons": dict(failure_reasons),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("PRM + failure-localization dataset build finished")
    print(f"Total examples: {summary['total_examples']}")
    print(f"Parse failures: {summary['parse_failures']}")
    print(f"Prediction-empty count: {summary['prediction_empty_count']}")
    print(f"Natural-failure examples kept: {summary['natural_failure_examples_kept']}")
    print(f"Fault-injection examples kept: {summary['fault_injection_examples_kept']}")
    print(f"Localization examples kept: {summary['localization_examples_kept']}")
    print(f"PRM train output: {prm_train_path}")
    print(f"PRM eval output: {prm_eval_path}")
    print(f"Localization eval output: {localization_path}")
    print(f"Summary: {summary_path}")
    print("=" * 70)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build split-first PRM + failure-localization datasets.")
    parser.add_argument(
        "--sources-config",
        type=str,
        default="",
        help="Optional JSON file listing multiple input sources.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/data/banlujie/RL-NL2SQL/eval_res/eval_results_voting_Qwen3.5-9B.json",
        help="Input JSON/JSONL eval file.",
    )
    parser.add_argument(
        "--tables-path",
        type=str,
        default="/data/banlujie/data/bird_dev/dev_tables.json",
        help="BIRD tables json path.",
    )
    parser.add_argument(
        "--db-root",
        type=str,
        default="/data/banlujie/data/bird_dev/dev_databases",
        help="Root directory of SQLite databases.",
    )
    parser.add_argument(
        "--template-path",
        type=str,
        default="/data/banlujie/RL-NL2SQL/templates/sql_generation_qwen.txt",
        help="Prompt template path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/banlujie/data/bird_dev/reward",
        help="Output directory for PRM and localization datasets.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio at original-example level.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit input rows; 0 means all.")
    parser.add_argument("--smoke", action="store_true", help="Enable smoke mode (small subset).")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.")
    parser.add_argument(
        "--sql-timeout",
        type=int,
        default=30,
        help="Per-SQL execution timeout in seconds for equivalence checks and mutation filtering.",
    )
    parser.add_argument(
        "--skip-timeout-repair",
        action="store_true",
        help="When the initial natural-failure equivalence check times out, skip earliest_repair and fall back to graph_surgery.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append newly built rows into an existing output_dir and merge build_summary statistics.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="Checkpoint path (default: <output_dir>/build_prm_failure_localization.ckpt.json).",
    )
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Save checkpoint every N examples.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if not (0.0 < cli_args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1).")
    build_pipeline(cli_args)
