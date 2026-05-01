import argparse
import json
import os
from collections import Counter
from typing import Dict, Iterable, Tuple


def _canonical_row_key(row: Dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _dedup_file(path: str, dry_run: bool) -> Tuple[int, int]:
    if not os.path.exists(path):
        return 0, 0

    seen = set()
    kept_rows = []
    total = 0

    for row in _iter_jsonl(path):
        total += 1
        key = _canonical_row_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept_rows.append(row)

    removed = total - len(kept_rows)
    if not dry_run and removed > 0:
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in kept_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)

    return total, removed


def _rebuild_summary(output_dir: str, dry_run: bool) -> Dict:
    summary_path = os.path.join(output_dir, "build_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}

    prm_train_path = os.path.join(output_dir, "v1_prm_train.jsonl")
    prm_eval_path = os.path.join(output_dir, "v1_prm_eval.jsonl")
    localization_path = os.path.join(output_dir, "failure_localization_eval.jsonl")

    split_counts = Counter()
    natural_count = 0
    fault_count = 0
    localization_count = 0

    for path in [prm_train_path, prm_eval_path]:
        if not os.path.exists(path):
            continue
        for row in _iter_jsonl(path):
            split = str(row.get("split", "")).strip()
            source_type = str(row.get("source_type", "")).strip()
            if source_type == "natural_failure":
                natural_count += 1
                if split:
                    split_counts[f"natural_failure_{split}"] += 1
            elif source_type == "fault_injection":
                fault_count += 1
                if split:
                    split_counts[f"fault_injection_{split}"] += 1

    if os.path.exists(localization_path):
        for _ in _iter_jsonl(localization_path):
            localization_count += 1

    output_rows = {
        "prm_train": sum(1 for _ in _iter_jsonl(prm_train_path)) if os.path.exists(prm_train_path) else 0,
        "prm_eval": sum(1 for _ in _iter_jsonl(prm_eval_path)) if os.path.exists(prm_eval_path) else 0,
        "failure_localization_eval": localization_count,
    }

    summary["output_rows"] = output_rows
    summary["split_rows"] = {
        "natural_failure_train": int(split_counts["natural_failure_train"]),
        "natural_failure_eval": int(split_counts["natural_failure_eval"]),
        "fault_injection_train": int(split_counts["fault_injection_train"]),
        "fault_injection_eval": int(split_counts["fault_injection_eval"]),
    }
    summary["natural_failure_examples_kept"] = natural_count
    summary["fault_injection_examples_kept"] = fault_count
    summary["localization_examples_kept"] = localization_count

    if not dry_run:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate PRM/failure-localization output JSONL files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/banlujie/data/bird_dev/reward",
        help="Directory containing v1_prm_train.jsonl, v1_prm_eval.jsonl, and failure_localization_eval.jsonl.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report duplicate counts without rewriting files.")
    args = parser.parse_args()

    targets = [
        "v1_prm_train.jsonl",
        "v1_prm_eval.jsonl",
        "failure_localization_eval.jsonl",
    ]

    print(f"Output dir: {args.output_dir}")
    for name in targets:
        path = os.path.join(args.output_dir, name)
        total, removed = _dedup_file(path, dry_run=args.dry_run)
        print(f"{name}: total={total}, duplicate_rows={removed}, unique_rows={total - removed}")

    summary = _rebuild_summary(args.output_dir, dry_run=args.dry_run)
    print("Updated summary output_rows:", json.dumps(summary.get("output_rows", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
