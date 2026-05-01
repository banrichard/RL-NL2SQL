import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple


PRM_SUFFIX_MARKER = "\nGenerated SQL Prefix Context:\n"


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _base_sft_prompt_from_prm_prompt(prm_prompt: str) -> str:
    marker_idx = prm_prompt.find(PRM_SUFFIX_MARKER)
    if marker_idx >= 0:
        return prm_prompt[:marker_idx].rstrip()
    return prm_prompt.rstrip()


def _reset_output_files(paths: Iterable[str]) -> None:
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)


def _pairwise_row_from_prm(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
        "margin": 1.0,
        "question_id": row.get("question_id"),
        "db_id": row.get("db_id"),
        "question": row.get("question"),
        "target_clause": row.get("target_clause"),
        "source_type": row.get("source_type"),
        "metadata": row.get("metadata", {}),
    }


def _pointwise_clause_rows_from_prm(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_meta = dict(row.get("metadata", {}))
    common = {
        "question_id": row.get("question_id"),
        "db_id": row.get("db_id"),
        "question": row.get("question"),
        "prompt": row.get("prompt"),
        "target_clause": row.get("target_clause"),
        "source_type": row.get("source_type"),
    }
    return [
        {
            **common,
            "response": row.get("chosen"),
            "score": 0.0,
            "label_type": "correct_clause",
            "metadata": {**base_meta, "is_positive": True},
        },
        {
            **common,
            "response": row.get("rejected"),
            "score": -1.0,
            "label_type": "incorrect_clause",
            "metadata": {**base_meta, "is_positive": False},
        },
    ]


def _exec_rm_pairwise_row_from_prm(row: Dict[str, Any]) -> Dict[str, Any]:
    base_prompt = _base_sft_prompt_from_prm_prompt(str(row.get("prompt", "")))
    base_meta = dict(row.get("metadata", {}))
    return {
        "prompt": base_prompt,
        "chosen": row.get("gt_sql", ""),
        "rejected": row.get("pred_sql", ""),
        "margin": 1.0,
        "question_id": row.get("question_id"),
        "db_id": row.get("db_id"),
        "question": row.get("question"),
        "source_type": row.get("source_type"),
        "metadata": {
            **base_meta,
            "label_scheme": "terminal_execution_pairwise",
            "chosen_exec_label": 1,
            "rejected_exec_label": 0,
        },
    }


def _exec_rm_pointwise_rows_from_prm(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_prompt = _base_sft_prompt_from_prm_prompt(str(row.get("prompt", "")))
    base_meta = dict(row.get("metadata", {}))
    common = {
        "question_id": row.get("question_id"),
        "db_id": row.get("db_id"),
        "question": row.get("question"),
        "prompt": base_prompt,
        "source_type": row.get("source_type"),
    }
    return [
        {
            **common,
            "response": row.get("gt_sql", ""),
            "score": 1.0,
            "label_type": "exec_correct",
            "metadata": {**base_meta, "is_positive": True},
        },
        {
            **common,
            "response": row.get("pred_sql", ""),
            "score": 0.0,
            "label_type": "exec_incorrect",
            "metadata": {**base_meta, "is_positive": False},
        },
    ]


def _register(counter: Counter, key: str, delta: int = 1) -> None:
    counter[key] += delta


def _dest_paths(output_dir: str) -> Dict[str, Dict[str, str]]:
    variants = {
        "full_pairwise": ("v1_prm_train.jsonl", "v1_prm_eval.jsonl"),
        "wo_fault_injection_pairwise": ("v1_prm_train.jsonl", "v1_prm_eval.jsonl"),
        "wo_counterfactual_pairwise": ("v1_prm_train.jsonl", "v1_prm_eval.jsonl"),
        "wo_topology_pointwise": ("train.jsonl", "eval.jsonl"),
        "exec_rm_pairwise": ("train.jsonl", "eval.jsonl"),
        "exec_rm_pointwise": ("train.jsonl", "eval.jsonl"),
    }
    out: Dict[str, Dict[str, str]] = {}
    for name, (train_name, eval_name) in variants.items():
        base = os.path.join(output_dir, name)
        out[name] = {
            "train": os.path.join(base, train_name),
            "eval": os.path.join(base, eval_name),
        }
    return out


def build_datasets(input_dir: str, output_dir: str) -> Dict[str, Any]:
    src_train = os.path.join(input_dir, "v1_prm_train.jsonl")
    src_eval = os.path.join(input_dir, "v1_prm_eval.jsonl")
    for path in [src_train, src_eval]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}")

    dests = _dest_paths(output_dir)
    _reset_output_files(
        path
        for variant in dests.values()
        for path in variant.values()
    )

    summary: Dict[str, Any] = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "variants": defaultdict(lambda: {"train": 0, "eval": 0}),
        "input_source_type_counts": defaultdict(lambda: {"train": 0, "eval": 0}),
        "input_source_dataset_counts": defaultdict(lambda: {"train": 0, "eval": 0}),
    }

    def _process_split(split_name: str, src_path: str) -> None:
        buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        rows_seen = 0
        for row in _iter_jsonl(src_path):
            rows_seen += 1
            source_type = str(row.get("source_type", "")).strip()
            source_dataset = str(row.get("metadata", {}).get("source_dataset", "")).strip() or "unknown"
            _register(summary["input_source_type_counts"][source_type], split_name)
            _register(summary["input_source_dataset_counts"][source_dataset], split_name)

            buckets[("full_pairwise", split_name)].append(_pairwise_row_from_prm(row))
            _register(summary["variants"]["full_pairwise"], split_name)

            if source_type == "natural_failure":
                buckets[("wo_fault_injection_pairwise", split_name)].append(_pairwise_row_from_prm(row))
                _register(summary["variants"]["wo_fault_injection_pairwise"], split_name)
            if source_type == "fault_injection":
                buckets[("wo_counterfactual_pairwise", split_name)].append(_pairwise_row_from_prm(row))
                _register(summary["variants"]["wo_counterfactual_pairwise"], split_name)

            topo_rows = _pointwise_clause_rows_from_prm(row)
            buckets[("wo_topology_pointwise", split_name)].extend(topo_rows)
            _register(summary["variants"]["wo_topology_pointwise"], split_name, len(topo_rows))

            exec_pair = _exec_rm_pairwise_row_from_prm(row)
            buckets[("exec_rm_pairwise", split_name)].append(exec_pair)
            _register(summary["variants"]["exec_rm_pairwise"], split_name)

            exec_point_rows = _exec_rm_pointwise_rows_from_prm(row)
            buckets[("exec_rm_pointwise", split_name)].extend(exec_point_rows)
            _register(summary["variants"]["exec_rm_pointwise"], split_name, len(exec_point_rows))

        for (variant_name, split_key), rows in buckets.items():
            _append_jsonl(dests[variant_name][split_key], rows)

        summary.setdefault("input_rows", {})[split_name] = rows_seen

    _process_split("train", src_train)
    _process_split("eval", src_eval)

    # Normalize defaultdicts for JSON serialization.
    summary["variants"] = {k: dict(v) for k, v in summary["variants"].items()}
    summary["input_source_type_counts"] = {
        k: dict(v) for k, v in summary["input_source_type_counts"].items()
    }
    summary["input_source_dataset_counts"] = {
        k: dict(v) for k, v in summary["input_source_dataset_counts"].items()
    }
    summary["definition_notes"] = {
        "wo_fault_injection_pairwise": "Keep only source_type == natural_failure from clause-level PRM pairs.",
        "wo_counterfactual_pairwise": "Keep only source_type == fault_injection from clause-level PRM pairs.",
        "wo_topology_pointwise": "Convert each clause-level PRM pair into two pointwise rows: chosen score 0.0, rejected score -1.0.",
        "exec_rm_pairwise": "Terminal execution supervision in pairwise form: full gt_sql as chosen, full pred_sql as rejected.",
        "exec_rm_pointwise": "Terminal execution supervision in pointwise form: gt_sql score 1.0, pred_sql score 0.0.",
    }

    summary_path = os.path.join(output_dir, "ablation_dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare failure-localization RM ablation datasets.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data/banlujie/data/bird_dev/reward",
        help="Directory containing v1_prm_train.jsonl and v1_prm_eval.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/banlujie/data/bird_dev/reward_ablation",
        help="Directory to write prepared ablation datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_datasets(args.input_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
