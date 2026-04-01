#!/usr/bin/env python3
"""
Deduplicate and split v1_prm_dataset.jsonl into train / eval splits.

Dedup key  : (prompt, chosen, rejected) — exact string match after whitespace collapse
DB split   : extract db_id from the "Database ID:\n<db_id>" pattern in prompt;
             randomly assign 10 % of unique dbs to eval (no db overlap with train).
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


DB_ID_PATTERN = re.compile(r"Database ID:\s*\n(\S+)", re.IGNORECASE)
EVAL_DB_RATIO  = 0.10   # fraction of unique dbs held out for eval
RANDOM_SEED    = 42


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


def extract_db_id(prompt: str) -> str:
    m = DB_ID_PATTERN.search(prompt)
    return m.group(1).strip() if m else "__unknown__"


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] line {lineno} JSON error: {e}")
    return rows


def write_jsonl(path: str, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):,} rows → {path}")


def main():
    parser = argparse.ArgumentParser(description="Dedup + DB-aware train/eval split for PRM dataset")
    parser.add_argument("--input",  default="/data/banlujie/data/bird_dev/reward/v1_prm_dataset.jsonl")
    parser.add_argument("--train",  default="/data/banlujie/data/bird_dev/reward/v1_prm_train.jsonl")
    parser.add_argument("--eval",   default="/data/banlujie/data/bird_dev/reward/v1_prm_eval.jsonl")
    parser.add_argument("--eval_db_ratio", type=float, default=EVAL_DB_RATIO,
                        help="Fraction of unique db_ids reserved for eval (default 0.10)")
    parser.add_argument("--seed",   type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ------------------------------------------------------------------ #
    # 1. Load
    # ------------------------------------------------------------------ #
    print(f"Loading {args.input} …")
    rows = load_jsonl(args.input)
    print(f"  Raw rows: {len(rows):,}")

    # ------------------------------------------------------------------ #
    # 2. Deduplicate  (key = collapsed prompt + chosen + rejected)
    # ------------------------------------------------------------------ #
    seen        = set()
    unique_rows = []
    for r in rows:
        key = (
            _squash(r.get("prompt",   "")),
            _squash(r.get("chosen",   "")),
            _squash(r.get("rejected", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        # Attach db_id for grouping (not written to output)
        r["_db_id"] = extract_db_id(r.get("prompt", ""))
        unique_rows.append(r)

    dup_removed = len(rows) - len(unique_rows)
    print(f"  After dedup: {len(unique_rows):,}  (removed {dup_removed:,} duplicates)")

    # ------------------------------------------------------------------ #
    # 3. Group by db_id
    # ------------------------------------------------------------------ #
    db_groups: dict = defaultdict(list)
    for r in unique_rows:
        db_groups[r["_db_id"]].append(r)

    all_dbs    = sorted(db_groups.keys())
    n_eval_dbs = max(1, round(len(all_dbs) * args.eval_db_ratio))
    rng.shuffle(all_dbs)
    eval_dbs   = set(all_dbs[:n_eval_dbs])
    train_dbs  = set(all_dbs[n_eval_dbs:])

    print(f"\n  Unique db_ids  : {len(all_dbs):,}")
    print(f"  Train db_ids   : {len(train_dbs):,}")
    print(f"  Eval  db_ids   : {len(eval_dbs):,}")
    print(f"  Eval  dbs      : {sorted(eval_dbs)}")

    # ------------------------------------------------------------------ #
    # 4. Split rows  (strip internal _db_id before writing)
    # ------------------------------------------------------------------ #
    train_rows = []
    eval_rows  = []
    for r in unique_rows:
        db  = r.pop("_db_id")
        out = {k: v for k, v in r.items()}   # clean copy
        if db in eval_dbs:
            eval_rows.append(out)
        else:
            train_rows.append(out)

    # Shuffle both splits for good measure
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    print(f"\n  Train rows : {len(train_rows):,}")
    print(f"  Eval  rows : {len(eval_rows):,}")

    # ------------------------------------------------------------------ #
    # 5. Write
    # ------------------------------------------------------------------ #
    write_jsonl(args.train, train_rows)
    write_jsonl(args.eval,  eval_rows)

    # ------------------------------------------------------------------ #
    # 6. Sanity check − no db overlap
    # ------------------------------------------------------------------ #
    def get_dbs(path):
        dbs = set()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    dbs.add(extract_db_id(r.get("prompt", "")))
        return dbs

    train_db_set = get_dbs(args.train)
    eval_db_set  = get_dbs(args.eval)
    overlap      = train_db_set & eval_db_set
    print(f"\n  Sanity check — DB overlap between train & eval: {len(overlap)}")
    if overlap:
        print(f"  [WARNING] overlapping dbs: {overlap}")
    else:
        print("  ✓ No DB overlap — split is clean.")

    print("\nDone.")


if __name__ == "__main__":
    main()
