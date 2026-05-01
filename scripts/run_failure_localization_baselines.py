import argparse
import csv
import importlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from build_v1_reward_dataset import extract_prefix_context, perform_graph_surgery
from dataset.build_ast_counterfactual_dataset import db_path_for, parse_sql
from dataset.expand_prm_dataset import CLAUSE_ORDER, clause_expression_map


DEFAULT_DATASET_PATH = "/data/banlujie/data/bird_dev/reward/failure_localization_eval.jsonl"
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "failure_localization_baselines.json")
DEFAULT_OUTPUT_PREFIX = os.path.join(REPO_ROOT, "eval_res", "failure_localization_results")

ORACLE_METHODS = {"Gold-First-Diff"}


@dataclass
class MethodRun:
    name: str
    status: str
    reason: str
    metrics: Optional[Dict[str, float]]
    n_examples: int
    failures: int
    per_seed_metrics: Optional[List[Dict[str, Any]]] = None
    breakdowns: Optional[Dict[str, Any]] = None
    prediction_path: Optional[str] = None
    oracle_only: bool = False


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_dataset(path: str, max_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            clauses = obj.get("clauses")
            prefixes = obj.get("prefix_contexts")
            idx = obj.get("root_cause_clause_index")
            if not isinstance(clauses, list) or not clauses:
                continue
            if not isinstance(prefixes, list):
                prefixes = []
            if len(prefixes) != len(clauses):
                prefixes = ["" for _ in clauses]
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= len(clauses):
                continue
            obj["prefix_contexts"] = prefixes
            rows.append(obj)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def _get_source_dataset(ex: Dict[str, Any]) -> str:
    meta = ex.get("metadata")
    if isinstance(meta, dict):
        source = str(meta.get("source_dataset", "")).strip()
        if source:
            return source
    source = str(ex.get("source_dataset", "")).strip()
    if source:
        return source
    return "default"


def _normalize_db_roots(value: Any) -> List[str]:
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            item = str(item).strip()
            if item:
                out.append(item)
        return out
    return []


def _resolve_db_path_with_roots(db_id: str, db_roots: List[str]) -> str:
    db_id = str(db_id or "").strip()
    if not db_id:
        return ""
    for root in db_roots:
        root = str(root or "").strip()
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


def _resolve_db_path_for_example(ex: Dict[str, Any], db_roots_by_source: Dict[str, List[str]], default_db_root: str) -> str:
    source_dataset = _get_source_dataset(ex)
    db_id = str(ex.get("db_id", "")).strip()
    roots = list(db_roots_by_source.get(source_dataset, []))
    if not roots and default_db_root:
        roots = [default_db_root]

    resolved = _resolve_db_path_with_roots(db_id, roots)
    if resolved:
        return resolved

    if default_db_root and default_db_root not in roots:
        resolved = _resolve_db_path_with_roots(db_id, [default_db_root])
        if resolved:
            return resolved
    return ""


def _derive_clause_names(pred_sql: str, clauses: List[str]) -> List[str]:
    root = parse_sql(pred_sql)
    if root is None:
        return [f"CLAUSE_{i}" for i in range(len(clauses))]
    cmap = clause_expression_map(root)
    names = [k for k in CLAUSE_ORDER if k in cmap]
    if len(names) == len(clauses):
        return names
    # fallback: align by exact clause text when possible
    text_to_name = {cmap[k].strip(): k for k in names}
    out: List[str] = []
    for i, c in enumerate(clauses):
        out.append(text_to_name.get(str(c).strip(), f"CLAUSE_{i}"))
    return out


def _normalize_ranking(ranking: List[int], n: int) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in ranking:
        if isinstance(x, int) and 0 <= x < n and x not in seen:
            out.append(x)
            seen.add(x)
    for i in range(n):
        if i not in seen:
            out.append(i)
    return out


def _rank_to_metrics(ranks_1based: List[int]) -> Dict[str, float]:
    n = len(ranks_1based)
    if n == 0:
        return {"Acc_loc": 0.0, "Hit@3": 0.0, "MRR": 0.0}
    acc = sum(1 for r in ranks_1based if r == 1) / n
    hit3 = sum(1 for r in ranks_1based if r <= 3) / n
    mrr = sum(1.0 / r for r in ranks_1based if r > 0) / n
    return {"Acc_loc": acc, "Hit@3": hit3, "MRR": mrr}


def _clause_count_bucket(n: int) -> str:
    if n <= 3:
        return "<=3"
    if n <= 5:
        return "4-5"
    return "6+"


def _compute_breakdowns(pred_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        return _rank_to_metrics([int(r["rank"]) for r in rows])

    breakdowns: Dict[str, Any] = {"by_source_type": {}, "by_clause_count_bucket": {}, "by_root_clause": {}}
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    by_bucket: Dict[str, List[Dict[str, Any]]] = {}
    by_root: Dict[str, List[Dict[str, Any]]] = {}

    for row in pred_rows:
        by_source.setdefault(str(row.get("source_type", "unknown")), []).append(row)
        by_bucket.setdefault(_clause_count_bucket(int(row.get("num_clauses", 0))), []).append(row)
        by_root.setdefault(str(row.get("root_cause_clause_name", "unknown")), []).append(row)

    breakdowns["by_source_type"] = {k: _agg(v) for k, v in by_source.items()}
    breakdowns["by_clause_count_bucket"] = {k: _agg(v) for k, v in by_bucket.items()}
    breakdowns["by_root_clause"] = {k: _agg(v) for k, v in by_root.items()}
    return breakdowns


def _write_predictions(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class SelfDebugJudge:
    def __init__(self, model_path: str, max_new_tokens: int, temperature: float):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        self.model.eval()

    @staticmethod
    def _extract_ranking(text: str) -> Optional[List[int]]:
        text = text.strip()
        candidates = [text]
        brace = re.findall(r"\{[\s\S]*\}", text)
        candidates.extend(brace)
        for cand in candidates:
            try:
                obj = json.loads(cand)
                ranking = obj.get("ranking")
                if isinstance(ranking, list):
                    out = [int(x) for x in ranking]
                    return out
            except Exception:
                continue
        list_match = re.search(r"\[(.*?)\]", text, flags=re.DOTALL)
        if list_match:
            nums = re.findall(r"-?\d+", list_match.group(1))
            if nums:
                return [int(x) for x in nums]
        return None

    def rank(self, ex: Dict[str, Any], clause_names: List[str]) -> Tuple[List[int], str]:
        self._load()
        assert self.tokenizer is not None and self.model is not None

        clauses = ex["clauses"]
        clause_block = "\n".join(
            f"{i}: [{clause_names[i]}] {clauses[i]}" for i in range(len(clauses))
        )
        prompt = (
            "You are a SQL debugging judge. Rank clause indices from most likely faulty to least likely faulty.\n"
            "Return strict JSON only: {\"ranking\": [indices...]}\n"
            "Use each index exactly once.\n\n"
            f"Question:\n{ex.get('question', '')}\n\n"
            f"Schema / Prompt Context:\n{ex.get('prompt', '')}\n\n"
            f"Predicted SQL:\n{ex.get('pred_sql', '')}\n\n"
            f"Gold SQL:\n{ex.get('gt_sql', '')}\n\n"
            f"Clauses:\n{clause_block}\n"
        )
        attempts = [
            prompt,
            prompt + "\nYour previous answer was invalid. Output ONLY valid JSON with full ranking.",
        ]
        for idx, cur_prompt in enumerate(attempts):
            enc = self.tokenizer(cur_prompt, return_tensors="pt")
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
            )
            txt = self.tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
            ranking = self._extract_ranking(txt)
            if ranking is not None:
                return ranking, f"ok_attempt_{idx + 1}"
        return list(range(len(clauses))), "parse_failed_fallback"


def _prepare_prm_module(rm_path: str, rm_max_len: int, device: str):
    rf = importlib.import_module("algorithm.reward_func")
    rf._RM_PATH = rm_path
    rf._RM_MAX_LEN = int(rm_max_len)
    os.environ["REWARD_MODEL_DEVICE"] = device
    rf._rm_model = None
    rf._rm_tokenizer = None
    rf._rm_device = None
    return rf


def _run_single_baseline(
    baseline: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    global_seed: int,
    db_root: str,
    db_roots_by_source: Dict[str, List[str]],
    output_prefix: str,
) -> MethodRun:
    name = baseline["name"]
    btype = baseline["type"]
    optional = bool(baseline.get("optional", False))
    oracle_only = bool(baseline.get("oracle_only", False) or name in ORACLE_METHODS)
    pred_path = f"{output_prefix}_predictions_{_slug(name)}.jsonl"
    failures = 0

    if btype == "self_debug":
        model_path = str(baseline.get("model_path", "")).strip()
        if not model_path:
            return MethodRun(name, "skipped", "missing_model_path", None, 0, 0, prediction_path=None, oracle_only=oracle_only)
        if not os.path.exists(model_path):
            return MethodRun(name, "skipped", f"model_path_not_found:{model_path}", None, 0, 0, prediction_path=None, oracle_only=oracle_only)
        try:
            judge = SelfDebugJudge(
                model_path=model_path,
                max_new_tokens=int(baseline.get("max_new_tokens", 192)),
                temperature=float(baseline.get("temperature", 0.0)),
            )
        except Exception as e:
            return MethodRun(name, "skipped", f"judge_init_failed:{type(e).__name__}", None, 0, 0, prediction_path=None, oracle_only=oracle_only)
    else:
        judge = None

    rf = None
    if btype == "prm":
        rm_path = str(baseline.get("rm_path", "")).strip()
        if not rm_path:
            status = "skipped" if optional else "failed"
            reason = "missing_rm_path"
            return MethodRun(name, status, reason, None, 0, 0, prediction_path=None, oracle_only=oracle_only)
        if not os.path.exists(rm_path):
            status = "skipped" if optional else "failed"
            reason = f"rm_path_not_found:{rm_path}"
            return MethodRun(name, status, reason, None, 0, 0, prediction_path=None, oracle_only=oracle_only)
        try:
            rf = _prepare_prm_module(
                rm_path=rm_path,
                rm_max_len=int(baseline.get("rm_max_len", 2048)),
                device=str(baseline.get("device", "auto")),
            )
        except Exception as e:
            status = "skipped" if optional else "failed"
            return MethodRun(name, status, f"prm_init_failed:{type(e).__name__}", None, 0, 0, prediction_path=None, oracle_only=oracle_only)

    if btype == "exec_rm":
        try:
            rf = importlib.import_module("algorithm.reward_func")
        except Exception as e:
            status = "skipped" if optional else "failed"
            return MethodRun(name, status, f"exec_rm_import_failed:{type(e).__name__}", None, 0, 0, prediction_path=None, oracle_only=oracle_only)

    if btype == "random":
        seeds = baseline.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            seeds = [global_seed + i for i in range(5)]
        per_seed_metrics: List[Dict[str, Any]] = []
        sample_predictions: List[Dict[str, Any]] = []
        for seed in seeds:
            rng = random.Random(int(seed))
            rows_this_seed: List[Dict[str, Any]] = []
            ranks: List[int] = []
            for ex in dataset:
                n = len(ex["clauses"])
                ranking = list(range(n))
                rng.shuffle(ranking)
                ranking = _normalize_ranking(ranking, n)
                true_idx = int(ex["root_cause_clause_index"])
                rank = ranking.index(true_idx) + 1
                ranks.append(rank)
                rows_this_seed.append(
                    {
                        "seed": int(seed),
                        "question_id": ex.get("question_id"),
                        "db_id": ex.get("db_id"),
                        "source_type": ex.get("source_type"),
                        "root_cause_clause_name": ex.get("root_cause_clause_name"),
                        "num_clauses": n,
                        "true_index": true_idx,
                        "ranking": ranking,
                        "rank": rank,
                    }
                )
            per_seed_metrics.append({"seed": int(seed), **_rank_to_metrics(ranks)})
            if not sample_predictions:
                sample_predictions = rows_this_seed

        mean_metrics = {
            "Acc_loc": sum(x["Acc_loc"] for x in per_seed_metrics) / len(per_seed_metrics),
            "Hit@3": sum(x["Hit@3"] for x in per_seed_metrics) / len(per_seed_metrics),
            "MRR": sum(x["MRR"] for x in per_seed_metrics) / len(per_seed_metrics),
        }
        _write_predictions(pred_path, sample_predictions)
        return MethodRun(
            name=name,
            status="completed",
            reason="ok",
            metrics=mean_metrics,
            n_examples=len(dataset),
            failures=0,
            per_seed_metrics=per_seed_metrics,
            breakdowns=_compute_breakdowns(sample_predictions),
            prediction_path=pred_path,
            oracle_only=oracle_only,
        )

    pred_rows: List[Dict[str, Any]] = []
    ranks_1based: List[int] = []
    pbar = tqdm(dataset, desc=f"Running {name}", leave=False)
    for ex in pbar:
        n = len(ex["clauses"])
        true_idx = int(ex["root_cause_clause_index"])
        pred_sql = str(ex.get("pred_sql", ""))
        gt_sql = str(ex.get("gt_sql", ""))
        clause_names = _derive_clause_names(pred_sql, ex["clauses"])
        ranking: List[int] = list(range(n))
        status_note = "ok"
        extra: Dict[str, Any] = {}

        try:
            if btype == "last_clause":
                ranking = [n - 1] + list(range(n - 2, -1, -1))
            elif btype == "gold_first_diff":
                surgery = perform_graph_surgery(pred_sql, gt_sql)
                if surgery is None:
                    status_note = "graph_surgery_failed_fallback"
                else:
                    c_name = str(surgery.get("divergent_clause", "")).strip()
                    if c_name in clause_names:
                        best = clause_names.index(c_name)
                        ranking = [best] + [i for i in range(n) if i != best]
                    else:
                        status_note = "divergent_clause_not_found_fallback"
            elif btype == "self_debug":
                assert judge is not None
                ranking_raw, note = judge.rank(ex, clause_names)
                ranking = ranking_raw
                status_note = note
            elif btype == "exec_rm":
                assert rf is not None
                db_path = _resolve_db_path_for_example(ex, db_roots_by_source, db_root)
                if not os.path.exists(db_path):
                    status_note = "db_not_found_fallback"
                else:
                    pred_root = parse_sql(pred_sql)
                    gt_root = parse_sql(gt_sql)
                    if pred_root is None or gt_root is None:
                        status_note = "parse_failed_fallback"
                    else:
                        pred_map = clause_expression_map(pred_root)
                        gt_map = clause_expression_map(gt_root)
                        scores = []
                        for i, c_name in enumerate(clause_names):
                            patched = dict(pred_map)
                            if c_name in gt_map:
                                patched[c_name] = gt_map[c_name]
                            elif c_name in patched:
                                patched.pop(c_name, None)
                            cand_sql = "\n".join(patched[c] for c in CLAUSE_ORDER if c in patched)
                            reward, exec_status = rf._execute_match_reward(db_path, cand_sql, gt_sql)
                            scores.append((i, float(reward), exec_status))
                        scores.sort(key=lambda x: (x[1], -x[0]), reverse=True)
                        ranking = [i for i, _s, _st in scores]
                        extra["exec_scores"] = scores
            elif btype == "prm":
                assert rf is not None
                sft_prompt = str(ex.get("prompt", "")).strip()
                if not sft_prompt:
                    status_note = "missing_prompt_fallback"
                else:
                    inputs: List[str] = []
                    for i, clause in enumerate(ex["clauses"]):
                        prefix = ""
                        if i < len(ex["prefix_contexts"]):
                            prefix = str(ex["prefix_contexts"][i] or "")
                        if not prefix:
                            c_name = clause_names[i] if i < len(clause_names) else "SELECT"
                            prefix = extract_prefix_context(pred_sql, c_name)
                        inputs.append(rf._build_prm_input(sft_prompt, prefix, str(clause)))
                    raw_scores = rf._score_texts(inputs) if inputs else []
                    signed_scores = [rf._signed_sigmoid_score(x) for x in raw_scores]
                    fault_scores = [-s for s in signed_scores]
                    ranking = [i for i, _ in sorted(enumerate(fault_scores), key=lambda x: x[1], reverse=True)]
                    extra["raw_scores"] = raw_scores
                    extra["signed_scores"] = signed_scores
                    extra["fault_scores"] = fault_scores
            else:
                return MethodRun(name, "failed", f"unknown_baseline_type:{btype}", None, 0, 0, prediction_path=None, oracle_only=oracle_only)
        except Exception as e:
            failures += 1
            status_note = f"error_fallback:{type(e).__name__}"

        ranking = _normalize_ranking(ranking, n)
        rank = ranking.index(true_idx) + 1
        ranks_1based.append(rank)
        pred_rows.append(
            {
                "question_id": ex.get("question_id"),
                "db_id": ex.get("db_id"),
                "source_type": ex.get("source_type"),
                "root_cause_clause_name": ex.get("root_cause_clause_name"),
                "num_clauses": n,
                "true_index": true_idx,
                "ranking": ranking,
                "rank": rank,
                "status_note": status_note,
                **extra,
            }
        )

    metrics = _rank_to_metrics(ranks_1based)
    _write_predictions(pred_path, pred_rows)
    return MethodRun(
        name=name,
        status="completed",
        reason="ok",
        metrics=metrics,
        n_examples=len(pred_rows),
        failures=failures,
        breakdowns=_compute_breakdowns(pred_rows),
        prediction_path=pred_path,
        oracle_only=oracle_only,
    )


def _to_table_rows(runs: List[MethodRun]) -> List[Dict[str, Any]]:
    rows = []
    for r in runs:
        rows.append(
            {
                "Method": r.name,
                "Acc_loc": None if r.metrics is None else r.metrics["Acc_loc"],
                "Hit@3": None if r.metrics is None else r.metrics["Hit@3"],
                "MRR": None if r.metrics is None else r.metrics["MRR"],
                "Status": r.status if r.reason == "ok" else f"{r.status} ({r.reason})",
                "oracle_only": r.oracle_only,
            }
        )
    rows.sort(
        key=lambda x: (
            1 if x["Acc_loc"] is None else 0,
            0.0 if x["Acc_loc"] is None else -float(x["Acc_loc"]),
        )
    )
    return rows


def _fmt_metric(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.4f}"


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Method", "Acc_loc", "Hit@3", "MRR", "Status"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "Method": r["Method"],
                    "Acc_loc": _fmt_metric(r["Acc_loc"]),
                    "Hit@3": _fmt_metric(r["Hit@3"]),
                    "MRR": _fmt_metric(r["MRR"]),
                    "Status": r["Status"],
                }
            )


def _write_markdown(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "| Method | Acc_loc | Hit@3 | MRR | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['Method']} | {_fmt_metric(r['Acc_loc'])} | {_fmt_metric(r['Hit@3'])} | {_fmt_metric(r['MRR'])} | {r['Status']} |"
        )
    lines.append("")
    lines.append("Note: `Gold-First-Diff` is an oracle-only reference and is non-deployable.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_latex(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    deployable = [r for r in rows if not r["oracle_only"] and r["Acc_loc"] is not None]
    best_method = max(deployable, key=lambda x: x["Acc_loc"])["Method"] if deployable else None
    out = [
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Method & Acc\\_loc & Hit@3 & MRR & Status \\\\",
        "\\hline",
    ]
    for r in rows:
        method = r["Method"]
        if best_method and method == best_method:
            method = f"\\textbf{{{method}}}"
        out.append(
            f"{method} & {_fmt_metric(r['Acc_loc'])} & {_fmt_metric(r['Hit@3'])} & {_fmt_metric(r['MRR'])} & {r['Status']} \\\\"
        )
    out.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
            "% Note: Gold-First-Diff is an oracle-only reference and non-deployable.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def _print_terminal_table(rows: List[Dict[str, Any]]) -> None:
    header = f"{'Method':30} {'Acc_loc':>8} {'Hit@3':>8} {'MRR':>8}  Status"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['Method'][:30]:30} {_fmt_metric(r['Acc_loc']):>8} {_fmt_metric(r['Hit@3']):>8} {_fmt_metric(r['MRR']):>8}  {r['Status']}"
        )


def _load_existing_runs(raw_json_path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(raw_json_path):
        return {}
    try:
        payload = _load_json(raw_json_path)
        runs = payload.get("results", [])
        if not isinstance(runs, list):
            return {}
        return {str(r.get("method")): r for r in runs if isinstance(r, dict) and r.get("method")}
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run held-out clause-level failure-localization baselines.")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH, help="Failure-localization eval JSONL path.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Baseline config JSON path.")
    parser.add_argument("--only", type=str, default="", help="Comma-separated baseline names to run.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing raw metrics JSON if present.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned baselines, do not execute.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--max-samples", type=int, default=0, help="Max dataset rows; 0 means all.")
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX, help="Output prefix (without extension).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.config):
        print(f"ERROR: config not found: {args.config}")
        return 1
    config = _load_json(args.config)

    dataset_path = args.dataset_path or config.get("dataset_path", DEFAULT_DATASET_PATH)
    if not os.path.exists(dataset_path):
        print(f"ERROR: missing held-out localization dataset: {dataset_path}")
        print("Dataset-construction task must run first (build failure_localization_eval.jsonl).")
        return 0

    output_prefix = args.output_prefix or config.get("output_prefix", DEFAULT_OUTPUT_PREFIX)
    db_root = str(config.get("db_root", "/data/banlujie/data/bird_dev/dev_databases"))
    raw_roots_by_source = config.get("db_roots_by_source", {})
    db_roots_by_source: Dict[str, List[str]] = {}
    if isinstance(raw_roots_by_source, dict):
        for key, value in raw_roots_by_source.items():
            db_roots_by_source[str(key).strip()] = _normalize_db_roots(value)
    baselines = [b for b in config.get("baselines", []) if isinstance(b, dict) and b.get("enabled", True)]
    if args.only.strip():
        only = {x.strip() for x in args.only.split(",") if x.strip()}
        baselines = [b for b in baselines if b.get("name") in only]

    if not baselines:
        print("No enabled baselines to run.")
        return 0

    dataset = _load_dataset(dataset_path, args.max_samples)
    if not dataset:
        print(f"ERROR: dataset has no valid rows: {dataset_path}")
        return 1

    print(f"Dataset: {dataset_path}")
    print(f"Valid examples: {len(dataset)}")
    print("Planned baselines:")
    for b in baselines:
        print(f" - {b.get('name')} [{b.get('type')}]")
    if args.dry_run:
        return 0

    raw_json_path = f"{output_prefix}.json"
    csv_path = f"{output_prefix}.csv"
    md_path = f"{output_prefix}.md"
    tex_path = f"{output_prefix}.tex"

    existing = _load_existing_runs(raw_json_path) if args.resume else {}

    run_results: List[MethodRun] = []
    for b in baselines:
        name = str(b.get("name"))
        if args.resume and name in existing and existing[name].get("status") in {"completed", "skipped"}:
            prev = existing[name]
            run_results.append(
                MethodRun(
                    name=name,
                    status=str(prev.get("status", "completed")),
                    reason=str(prev.get("reason", "resumed")),
                    metrics=prev.get("metrics"),
                    n_examples=int(prev.get("n_examples", 0)),
                    failures=int(prev.get("failures", 0)),
                    per_seed_metrics=prev.get("per_seed_metrics"),
                    breakdowns=prev.get("breakdowns"),
                    prediction_path=prev.get("prediction_path"),
                    oracle_only=bool(prev.get("oracle_only", False)),
                )
            )
            continue

        start = time.time()
        result = _run_single_baseline(
            baseline=b,
            dataset=dataset,
            global_seed=args.seed,
            db_root=db_root,
            db_roots_by_source=db_roots_by_source,
            output_prefix=output_prefix,
        )
        elapsed = time.time() - start
        print(
            f"[{result.status}] {result.name} | reason={result.reason} | n={result.n_examples} | failures={result.failures} | {elapsed:.1f}s"
        )
        run_results.append(result)

    rows = _to_table_rows(run_results)
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    _write_latex(tex_path, rows)

    payload = {
        "meta": {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "dataset_path": dataset_path,
            "config_path": args.config,
            "output_prefix": output_prefix,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "n_examples": len(dataset),
        },
        "results": [
            {
                "method": r.name,
                "status": r.status,
                "reason": r.reason,
                "metrics": r.metrics,
                "n_examples": r.n_examples,
                "failures": r.failures,
                "per_seed_metrics": r.per_seed_metrics,
                "breakdowns": r.breakdowns,
                "prediction_path": r.prediction_path,
                "oracle_only": r.oracle_only,
            }
            for r in run_results
        ],
        "table_rows": rows,
    }
    _save_json(raw_json_path, payload)

    print("\nFinal summary:")
    _print_terminal_table(rows)
    print(f"\nOutputs:\n- {raw_json_path}\n- {csv_path}\n- {md_path}\n- {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
