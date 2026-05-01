"""
Clause-PRM as a Candidate Verifier experiment.

Selectors: majority_vote, random, exec_rm, self_rerank, clause_prm
Splits:    bird_dev, spider_dev, spider_test
"""
import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

import eval_bird
from database.sql_executor import SQLExecutionResultType, execute_sql_with_timeout
from dataset.utils import serialize_schema

RANDOM_SEEDS = [42, 123, 7]

SPLITS: Dict[str, dict] = {
    "bird_dev": {
        "files": {
            "gpt-5.4":                  "eval_res/eval_results_api_voting_bird_dev_gpt-5.4.json",
            "claude-sonnet-4-20250514": "eval_res/eval_results_api_voting_bird_dev_claude-sonnet-4-20250514.json",
            "claude-sonnet-4-6":        "eval_res/eval_results_api_voting_bird_dev_claude-sonnet-4-6.json",
            "gemini-2.5-pro":           "eval_res/eval_results_api_voting_bird_dev_gemini-2.5-pro.json",
        },
        "gold_field": "SQL",
        "db_root": "/data/banlujie/data/bird_dev/dev_databases",
        "tables_file": "/data/banlujie/data/bird_dev/dev_tables.json",
    },
    "spider_dev": {
        "files": {
            "gpt-5.4":                  "eval_res/eval_results_api_voting_spider_dev_gpt-5.4.json",
            "claude-sonnet-4-20250514": "eval_res/eval_results_api_voting_spider_dev_claude-sonnet-4-20250514.json",
            "claude-sonnet-4-6":        "eval_res/eval_results_api_voting_spider_dev_claude-sonnet-4-6.json",
            "gemini-2.5-pro":           "eval_res/eval_results_api_voting_spider_dev_gemini-2.5-pro.json",
        },
        "gold_field": "query",
        "db_root": "/data/banlujie/data/spider/database",
        "tables_file": "/data/banlujie/data/spider/tables.json",
    },
    "spider_test": {
        "files": {
            "gpt-5.4":                  "eval_res/eval_results_api_voting_spider_test_gpt-5.4.json",
            "claude-sonnet-4-20250514": "eval_res/eval_results_api_voting_spider_test_claude-sonnet-4-20250514.json",
            "claude-sonnet-4-6":        "eval_res/eval_results_api_voting_spider_test_claude-sonnet-4-6.json",
            "gemini-2.5-pro":           "eval_res/eval_results_api_voting_spider_test_gemini-2.5-pro.json",
        },
        "gold_field": "query",
        "db_root": "/data/banlujie/data/spider/test_database",
        "tables_file": "/data/banlujie/data/spider/test_tables.json",
    },
}


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

def load_schema_map(tables_file: str) -> Dict[str, dict]:
    with open(tables_file) as f:
        tables = json.load(f)
    return {t["db_id"]: t for t in tables}


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

def select_majority_vote(pred_sqls: List[str]) -> Tuple[int, str]:
    normalized = [s.strip().lower() for s in pred_sqls]
    counts = Counter(normalized)
    winner = counts.most_common(1)[0][0]
    idx = normalized.index(winner)
    return idx, pred_sqls[idx]


def select_random(pred_sqls: List[str], seed: int) -> Tuple[int, str]:
    idx = random.Random(seed).randint(0, len(pred_sqls) - 1)
    return idx, pred_sqls[idx]


def select_exec_rm(
    pred_sqls: List[str],
    gold_sql: str,
    db_path: str,
    exec_rm_model,
    tokenizer,
) -> Tuple[int, str]:
    """Score each candidate with the Exec-RM model (pointwise), pick argmax."""
    from algorithm.reward_func import _build_sft_prompt
    import torch

    inputs = [f"SQL: {sql}" for sql in pred_sqls]
    enc = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.to(exec_rm_model.device) for k, v in enc.items()}
    with torch.no_grad():
        scores = exec_rm_model(**enc).squeeze(-1).tolist()
    if isinstance(scores, float):
        scores = [scores]
    idx = scores.index(max(scores))
    return idx, pred_sqls[idx]


def _build_rerank_messages(item: dict, schema_str: str) -> List[dict]:
    sqls = item["pred_sqls"]
    candidates = "\n".join(
        f"Candidate {i+1}:\n```sql\n{sql}\n```" for i, sql in enumerate(sqls)
    )
    evidence_line = f"Evidence: {item['evidence']}\n" if item.get("evidence") else ""
    content = (
        f"You are an expert SQL evaluator.\n"
        f"Question: {item['question']}\n"
        f"{evidence_line}"
        f"Database Schema:\n{schema_str}\n\n"
        f"{candidates}\n\n"
        f"Reply with ONLY the candidate number (1 to {len(sqls)})."
    )
    return [{"role": "user", "content": content}]


async def _self_rerank_all(
    items: List[dict],
    schema_map: Dict[str, dict],
    model: str,
    base_url: str,
    api_key: str,
    concurrency: int,
) -> Tuple[List[Tuple[int, str]], int]:
    from infer_api import call_api

    semaphore = asyncio.Semaphore(concurrency)
    n_parse_fail = 0

    async with httpx.AsyncClient() as client:
        tasks = [
            call_api(
                client, semaphore, base_url, api_key, model,
                messages=_build_rerank_messages(item, serialize_schema(schema_map.get(item["db_id"], {}))),
                max_tokens=16, temperature=0.0,
                idx=i, candidate_idx=0,
            )
            for i, item in enumerate(items)
        ]
        raw = await asyncio.gather(*tasks)

    results = []
    for (idx, _, response), item in zip(raw, items):
        n = len(item["pred_sqls"])
        m = re.search(r"\b([1-9])\b", response.strip())
        chosen = int(m.group(1)) - 1 if m and 1 <= int(m.group(1)) <= n else 0
        if not m:
            n_parse_fail += 1
        results.append((chosen, item["pred_sqls"][chosen]))
    return results, n_parse_fail


def _score_clause_prm(items: List[dict], schema_map: Dict[str, dict]) -> List[Tuple[int, str]]:
    from algorithm.reward_func import (
        _build_prm_input, _build_sft_prompt, _decompose_sql, _load_rm, _score_texts,
    )
    _load_rm()

    selections = []
    for item in items:
        schema_str = serialize_schema(schema_map.get(item["db_id"], {}))
        sft_prompt = _build_sft_prompt(schema_str, item["question"])
        pred_sqls = item["pred_sqls"]

        all_inputs: List[str] = []
        spans: List[Tuple[int, int]] = []
        for sql in pred_sqls:
            clauses = _decompose_sql(sql)
            start = len(all_inputs)
            prefix = ""
            for _, clause_text in clauses:
                all_inputs.append(_build_prm_input(sft_prompt, prefix, clause_text))
                prefix = (prefix + " " + clause_text).strip()
            spans.append((start, len(all_inputs)))

        if not all_inputs:
            selections.append((0, pred_sqls[0]))
            continue

        all_scores = _score_texts(all_inputs)
        candidate_scores = [
            sum(all_scores[s:e]) / max(e - s, 1) for s, e in spans
        ]
        idx = candidate_scores.index(max(candidate_scores))
        selections.append((idx, pred_sqls[idx]))
    return selections


# ---------------------------------------------------------------------------
# EX computation
# ---------------------------------------------------------------------------

def compute_ex(
    items: List[dict],
    chosen_sqls: List[str],
    gold_field: str,
    db_root: str,
    num_cpus: int = 20,
    timeout: int = 30,
) -> float:
    eval_bird.evaluation_results = []
    db_files = [f"{db_root}/{item['db_id']}/{item['db_id']}.sqlite" for item in items]
    questions = [item["question"] for item in items]
    gold_sqls = [item[gold_field] for item in items]
    eval_bird.evaluate_sqls_parallel(
        db_files, questions, chosen_sqls, gold_sqls,
        num_cpus=num_cpus, timeout=timeout,
    )
    results = eval_bird.evaluation_results or []
    return sum(r["correctness"] for r in results) / len(items) if items else 0.0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, lines: List[dict]):
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _plot(rows: List[dict], out_path: Path):
    splits = ["bird_dev", "spider_dev", "spider_test"]
    selectors = ["MajorityVote", "Random_mean", "SelfRerank", "ClausePRM"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, split in zip(axes, splits):
        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            ax.set_title(f"{split} (no data)")
            continue
        generators = [r["generator"] for r in split_rows]
        x = list(range(len(generators)))
        width = 0.18
        for i, (sel, color) in enumerate(zip(selectors, colors)):
            vals = [r.get(sel, 0) * 100 for r in split_rows]
            offsets = [xi + (i - len(selectors) / 2 + 0.5) * width for xi in x]
            ax.bar(offsets, vals, width, label=sel, color=color)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(generators, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("EX Accuracy (%)")
        ax.set_title(split)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_run_log(out_dir: Path, args, rows: List[dict], missing: List[str], stats: dict):
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    lines = [
        "# Candidate Verifier Run Log",
        f"Date: {datetime.now().isoformat()}",
        f"Commit: {commit}",
        f"Command: python3 eval_candidate_verifier.py {' '.join(sys.argv[1:])}",
        f"Selectors: {args.selectors}",
        f"Self-rerank model: {args.self_rerank_model}",
        f"Clause-PRM ckpt: {args.rm_path}",
        f"Exec-RM ckpt: {args.exec_rm_path}",
        f"Seeds: {RANDOM_SEEDS}",
        f"Smoke test N: {args.smoke_test}",
        "",
        "## Missing files",
    ] + (missing or ["(none)"]) + [
        "",
        "## Failure stats",
    ] + [f"- {k}: {v}" for k, v in stats.items()] + [
        "",
        "## Results summary",
        "| split | generator | MV | ClausePRM | Delta |",
        "|---|---|---|---|---|",
    ] + [
        f"| {r['split']} | {r['generator']} | {r.get('MajorityVote',0):.4f} | {r.get('ClausePRM',0):.4f} | {r.get('Delta_MV_ClausePRM',0):+.4f} |"
        for r in rows
    ]
    (out_dir / "run_candidate_verifier.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["bird_dev", "spider_dev", "spider_test"])
    p.add_argument("--selectors", nargs="+",
                   default=["majority_vote", "random", "self_rerank", "clause_prm"])
    p.add_argument("--self_rerank_model", default="gpt-5.4")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--num_cpus", type=int, default=20)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--smoke_test", type=int, default=0)
    p.add_argument("--out_dir", default="results/candidate_verifier")
    p.add_argument("--rm_path", default="/data/banlujie/ckpt/qwen3-0.6b-rm")
    p.add_argument("--exec_rm_path", default="/data/banlujie/ckpt/qwen3-0.6b-rm-exec-pointwise")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("REWARD_MODEL_PATH", args.rm_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = os.environ.get("BASE_URL", "https://tiantianapi.com/v1")
    api_key = os.environ.get("API_KEY", "")

    # Load Exec-RM model if requested
    exec_rm_model = exec_rm_tokenizer = None
    if "exec_rm" in args.selectors:
        from openrlhf.models import get_llm_for_sequence_regression
        from transformers import AutoTokenizer
        print(f"[exec_rm] Loading {args.exec_rm_path}", flush=True)
        exec_rm_model = get_llm_for_sequence_regression(
            args.exec_rm_path,
            "reward",
            normalize_reward=False,
            attn_implementation="flash_attention_2",
            param_dtype="bf16",
            value_head_prefix="score",
            device_map="cuda:0",
        )
        exec_rm_tokenizer = AutoTokenizer.from_pretrained(args.exec_rm_path)

    all_rows: List[dict] = []
    raw_gen_lines: List[dict] = []
    pred_lines: List[dict] = []
    missing: List[str] = []
    stats: dict = {"self_rerank_parse_fail": 0, "skipped_items": 0}

    for split_name in args.splits:
        cfg = SPLITS[split_name]
        schema_map = load_schema_map(cfg["tables_file"])

        for generator, rel_path in cfg["files"].items():
            path = Path(rel_path)
            if not path.exists():
                msg = f"MISSING: {rel_path}"
                print(f"[SKIP] {msg}", flush=True)
                missing.append(msg)
                continue

            with open(path) as f:
                items = json.load(f)
            if args.smoke_test:
                items = items[: args.smoke_test]

            # filter items with empty pred_sqls
            valid = [it for it in items if it.get("pred_sqls")]
            stats["skipped_items"] += len(items) - len(valid)
            items = valid

            N = len(items[0]["pred_sqls"]) if items else 0
            print(f"[{split_name}/{generator}] {len(items)} items, N={N}", flush=True)

            # write raw_generations
            for i, item in enumerate(items):
                raw_gen_lines.append({
                    "split": split_name, "generator": generator, "item_idx": i,
                    "question_id": item.get("question_id"), "db_id": item["db_id"],
                    "question": item["question"],
                    "pred_sqls": item["pred_sqls"],
                    "gold_sql": item.get(cfg["gold_field"], ""),
                })

            row: dict = {"split": split_name, "generator": generator, "N": N}

            # --- Majority Vote ---
            mv_ex = None
            if "majority_vote" in args.selectors:
                mv_sqls = [select_majority_vote(it["pred_sqls"])[1] for it in items]
                mv_ex = compute_ex(items, mv_sqls, cfg["gold_field"], cfg["db_root"],
                                   args.num_cpus, args.timeout)
                row["MajorityVote"] = mv_ex
                _record_preds(pred_lines, items, mv_sqls, split_name, generator, "majority_vote",
                              [select_majority_vote(it["pred_sqls"])[0] for it in items])

            # --- Random ---
            if "random" in args.selectors:
                rand_exs = []
                for seed in RANDOM_SEEDS:
                    sqls = [select_random(it["pred_sqls"], seed)[1] for it in items]
                    rand_exs.append(compute_ex(items, sqls, cfg["gold_field"], cfg["db_root"],
                                               args.num_cpus, args.timeout))
                    row[f"Random_seed{seed}"] = rand_exs[-1]
                row["Random_mean"] = sum(rand_exs) / len(rand_exs)
                # record predictions for seed 42
                sqls42 = [select_random(it["pred_sqls"], 42)[1] for it in items]
                _record_preds(pred_lines, items, sqls42, split_name, generator, "random",
                              [select_random(it["pred_sqls"], 42)[0] for it in items])

            # --- Exec-RM ---
            if "exec_rm" in args.selectors and exec_rm_model is not None:
                execrm_sqls, execrm_idxs = [], []
                for it in items:
                    idx, sql = select_exec_rm(
                        it["pred_sqls"], it.get(cfg["gold_field"], ""),
                        f"{cfg['db_root']}/{it['db_id']}/{it['db_id']}.sqlite",
                        exec_rm_model, exec_rm_tokenizer,
                    )
                    execrm_sqls.append(sql)
                    execrm_idxs.append(idx)
                row["ExecRM"] = compute_ex(items, execrm_sqls, cfg["gold_field"], cfg["db_root"],
                                           args.num_cpus, args.timeout)
                _record_preds(pred_lines, items, execrm_sqls, split_name, generator, "exec_rm", execrm_idxs)

            # --- Self-Rerank ---
            if "self_rerank" in args.selectors:
                sr_results, n_fail = asyncio.run(_self_rerank_all(
                    items, schema_map, args.self_rerank_model,
                    base_url, api_key, args.concurrency,
                ))
                stats["self_rerank_parse_fail"] += n_fail
                sr_sqls = [sql for _, sql in sr_results]
                sr_idxs = [idx for idx, _ in sr_results]
                row["SelfRerank"] = compute_ex(items, sr_sqls, cfg["gold_field"], cfg["db_root"],
                                               args.num_cpus, args.timeout)
                _record_preds(pred_lines, items, sr_sqls, split_name, generator, "self_rerank", sr_idxs)

            # --- Clause-PRM ---
            if "clause_prm" in args.selectors:
                prm_results = _score_clause_prm(items, schema_map)
                prm_sqls = [sql for _, sql in prm_results]
                prm_idxs = [idx for idx, _ in prm_results]
                row["ClausePRM"] = compute_ex(items, prm_sqls, cfg["gold_field"], cfg["db_root"],
                                              args.num_cpus, args.timeout)
                _record_preds(pred_lines, items, prm_sqls, split_name, generator, "clause_prm", prm_idxs)

            # Delta vs MajorityVote
            for sel in ["ExecRM", "SelfRerank", "ClausePRM"]:
                if sel in row and mv_ex is not None:
                    row[f"Delta_MV_{sel}"] = row[sel] - mv_ex

            all_rows.append(row)
            mv_str = f"{mv_ex:.4f}" if mv_ex is not None else "N/A"
            print(f"  MV={mv_str}  ClausePRM={row.get('ClausePRM', 'N/A')}", flush=True)

    # Write outputs
    _write_jsonl(out_dir / "raw_generations.jsonl", raw_gen_lines)
    _write_jsonl(out_dir / "extracted_sql.jsonl", raw_gen_lines)  # pred_sqls already extracted
    _write_jsonl(out_dir / "selector_predictions.jsonl", pred_lines)

    _write_csv(out_dir / "metrics_by_split.csv", all_rows)
    _plot(all_rows, out_dir / "candidate_verifier_delta_plot.png")
    _write_run_log(out_dir, args, all_rows, missing, stats)

    print(f"\nResults written to {out_dir}/", flush=True)


def _record_preds(pred_lines, items, sqls, split, generator, selector, idxs):
    for i, (item, sql, idx) in enumerate(zip(items, sqls, idxs)):
        pred_lines.append({
            "split": split, "generator": generator, "item_idx": i,
            "selector": selector, "chosen_idx": idx, "chosen_sql": sql,
        })


def _write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    fieldnames = [
        "split", "generator", "N",
        "MajorityVote", "Random_mean", "Random_seed42", "Random_seed123", "Random_seed7",
        "ExecRM", "SelfRerank", "ClausePRM",
        "Delta_MV_ExecRM", "Delta_MV_SelfRerank", "Delta_MV_ClausePRM",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
