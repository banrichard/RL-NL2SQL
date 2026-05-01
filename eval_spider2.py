#!/usr/bin/env python3
"""
Spider2-Lite 评测脚本

数据库类型分布（官方 547 条）：
  bq/*   180 条  BigQuery（需 bigquery_credential.json）
  ga/*    25 条  BigQuery（同上）
  local/  135 条  本地 SQLite（无需凭证）
  sf_bq/  189 条  官方 evaluate.py 实际上也跳过（sf 前缀不匹配 bq/ga/local）
  sf/      18 条  同上

本脚本三种模式：

  --mode local_only（默认，无需凭证）
    只评测 135 条 local SQLite 样本，立即可用。
    输出 local 准确率和官方口径分数（分母 547）。

  --mode bq（推荐，需 bigquery_credential.json）
    在本地执行 bq/* + ga/* 的 BigQuery SQL，加上 local SQLite，
    共评测 180+25+135=340 条，覆盖官方可评测的全部样本。
    凭证路径：evaluation_suite/bigquery_credential.json（自动找到）

  --mode official
    直接调用官方 evaluate.py --mode sql，cwd 设为 evaluation_suite/，
    使得 bigquery_credential.json 相对路径自动生效。
    sf_bq/* 样本官方也跳过，结果与 --mode bq 等价。

输出：
  eval_res/spider2/spider2_eval_<model>_eval.json   逐条结果
  eval_res/logs/eval_stats_spider2_<model>_<ts>.log  summary log（与 BIRD eval 格式一致）
"""

import argparse
import datetime
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

TOTAL_OFFICIAL = 547

# instance_id 前缀 → 数据库引擎
def _db_engine(instance_id: str) -> str:
    if instance_id.startswith(("bq", "ga")):
        return "BigQuery"
    if instance_id.startswith("local"):
        return "SQLite"
    return "Skipped"  # sf_bq / sf 等，官方也跳过


# ── 结果比较（与官方 evaluate.py 完全对齐）───────────────────────────────────

def _norm(v):
    return 0 if pd.isna(v) else v


def _vec_match(v1, v2, tol=1e-2, ignore_order=False):
    v1, v2 = [_norm(x) for x in v1], [_norm(x) for x in v2]
    if ignore_order:
        k = lambda x: (x is None, str(x), isinstance(x, (int, float)))
        v1, v2 = sorted(v1, key=k), sorted(v2, key=k)
    if len(v1) != len(v2):
        return False
    for a, b in zip(v1, v2):
        if pd.isna(a) and pd.isna(b):
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), abs_tol=tol):
                return False
        elif a != b:
            return False
    return True


def _cmp(pred: pd.DataFrame, gold: pd.DataFrame, condition_cols=None, ignore_order=False) -> int:
    gc = gold.iloc[:, condition_cols] if condition_cols else gold
    tg, tp = gc.transpose().values.tolist(), pred.transpose().values.tolist()
    for gv in tg:
        if not any(_vec_match(gv, pv, ignore_order=ignore_order) for pv in tp):
            return 0
    return 1


def _cmp_multi(pred: pd.DataFrame, golds, condition_cols=None, ignore_order=False) -> int:
    if not golds:
        return 0
    if condition_cols in (None, [], [[]], [None]):
        condition_cols = [[] for _ in golds]
    elif len(golds) > 1 and not all(isinstance(s, list) for s in condition_cols):
        condition_cols = [condition_cols] * len(golds)
    for i, g in enumerate(golds):
        if _cmp(pred, g, condition_cols[i] or None, ignore_order):
            return 1
    return 0


@lru_cache(maxsize=None)
def _load_csv(p: str) -> pd.DataFrame:
    return pd.read_csv(p)


def _resolve_gold(iid: str, gold_dir: str) -> Tuple[List[Path], bool]:
    single = Path(gold_dir) / f"{iid}.csv"
    if single.exists():
        return [single], True
    pat = re.compile(rf"^{re.escape(iid)}(_[a-z])?\.csv$")
    files = sorted(f for f in os.listdir(gold_dir) if pat.match(f))
    return [Path(gold_dir) / f for f in files], False


# ── SQLite 执行（带 SIGALRM 超时）───────────────────────────────────────────

class _T(Exception):
    pass


def _exec_sqlite(db_path: str, sql: str, save_dir: str, fname: str, timeout: int) -> Tuple[bool, Optional[str]]:
    def _h(s, f):
        raise _T()
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(timeout)
    try:
        conn = sqlite3.connect(db_path)
        mem = sqlite3.connect(":memory:")
        conn.backup(mem)
        conn.close()
        try:
            os.makedirs(save_dir, exist_ok=True)
            first = True
            for chunk in pd.read_sql_query(sql, mem, chunksize=500):
                chunk.to_csv(os.path.join(save_dir, fname), mode="w" if first else "a", header=first, index=False)
                first = False
            return True, None
        except _T:
            return False, "timeout"
        except Exception as e:
            return False, _classify_err(str(e))
        finally:
            mem.close()
    except _T:
        return False, "timeout"
    except Exception as e:
        return False, f"connect_error:{e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _classify_err(msg: str) -> str:
    for kw, label in [("no such column", "no_such_column"), ("syntax error", "syntax_error"),
                      ("no such table", "no_such_table"), ("ambiguous", "ambiguous_column")]:
        if kw in msg.lower():
            return f"OperationalError:{label}"
    return f"OperationalError:{msg[:60]}"


def _find_sqlite(db_name: str, db_base: Path) -> Optional[Path]:
    for sub in ("spider2-localdb", "sqlite"):
        d = db_base / sub
        if d.is_dir():
            for f in d.iterdir():
                if f.name.lower() == f"{db_name.lower()}.sqlite":
                    return f
    return None


# ── BigQuery 执行 ─────────────────────────────────────────────────────────────

def _exec_bigquery(sql: str, save_dir: str, fname: str, credential_path: str, iid: str) -> Tuple[bool, Optional[str]]:
    try:
        from google.cloud import bigquery as bq
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
        client = bq.Client()
        job = client.query(sql)
        df = job.result().to_dataframe()
        if df.empty:
            df.to_csv(os.path.join(save_dir, fname), index=False)
            return True, None
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, fname), index=False)
        return True, None
    except Exception as e:
        return False, f"BigQueryError:{str(e)[:120]}"


# ── 单条评测 ──────────────────────────────────────────────────────────────────

def _eval_one(
    item: dict,
    eval_std: dict,
    gold_dir: str,
    db_base: Path,
    tmp_dir: Path,
    timeout: int,
    bq_credential: Optional[str],
    eval_bq: bool,
) -> dict:
    iid = item["instance_id"]
    sql = (item.get("pred_sql") or "").strip()
    std = eval_std.get(iid, {})
    cc = std.get("condition_cols") or None
    io = std.get("ignore_order", False)

    # sf_bq / sf 类：官方也跳过，直接标记
    engine = _db_engine(iid)
    if engine == "Skipped":
        return {"instance_id": iid, "score": -1, "pred_sql": sql,
                "error_info": "skipped:sf_prefix_not_supported_by_official_evaluate"}

    if not sql:
        return {"instance_id": iid, "score": 0, "pred_sql": sql, "error_info": "no_sql_generated"}

    tdir = tmp_dir / f"t{threading.get_ident()}_{iid}"
    tdir.mkdir(parents=True, exist_ok=True)
    fname = f"{iid}.csv"

    # BigQuery
    if engine == "BigQuery":
        if not eval_bq or not bq_credential:
            return {"instance_id": iid, "score": -1, "pred_sql": sql,
                    "error_info": "skipped:bigquery_credentials_not_provided"}
        ok, err = _exec_bigquery(sql, str(tdir), fname, bq_credential, iid)
        if not ok:
            return {"instance_id": iid, "score": 0, "pred_sql": sql, "error_info": f"pred_sql_error:{err}"}

    # SQLite
    else:
        db_name = item.get("db", "")
        db_path = _find_sqlite(db_name, db_base)
        if db_path is None:
            return {"instance_id": iid, "score": 0, "pred_sql": sql,
                    "error_info": f"sqlite_not_found:db={db_name}"}
        ok, err = _exec_sqlite(str(db_path), sql, str(tdir), fname, timeout)
        if not ok:
            return {"instance_id": iid, "score": 0, "pred_sql": sql, "error_info": f"pred_sql_error:{err}"}

    pred_pd = pd.read_csv(tdir / fname)
    gold_paths, is_single = _resolve_gold(iid, gold_dir)
    if not gold_paths:
        return {"instance_id": iid, "score": 0, "pred_sql": sql, "error_info": "no_gold_file"}

    try:
        score = (_cmp(pred_pd, _load_csv(str(gold_paths[0])), cc, io) if is_single
                 else _cmp_multi(pred_pd, [_load_csv(str(p)) for p in gold_paths], cc, io))
    except Exception as e:
        return {"instance_id": iid, "score": 0, "pred_sql": sql, "error_info": f"compare_error:{e}"}

    return {"instance_id": iid, "score": score, "pred_sql": sql,
            "error_info": None if score == 1 else "result_mismatch"}


# ── Summary（与 BIRD eval log 格式一致）──────────────────────────────────────

def _build_summary(results: List[dict], mode: str, pred_file: str) -> dict:
    evaluated = [r for r in results if r["score"] != -1]
    skipped = [r for r in results if r["score"] == -1]
    correct = sum(1 for r in evaluated if r["score"] == 1)
    total_eval = len(evaluated)

    exec_ok = sum(1 for r in evaluated if r["error_info"] in (None, "result_mismatch"))
    exec_fail = total_eval - exec_ok

    failure_reasons: Dict[str, int] = {}
    for r in evaluated:
        if r["score"] != 1:
            k = r.get("error_info") or "unknown"
            failure_reasons[k] = failure_reasons.get(k, 0) + 1

    failure_detail = {
        err: {
            "count": cnt,
            "ratio_in_failures": cnt / max(total_eval - correct, 1),
            "ratio_in_evaluated": cnt / max(total_eval, 1),
        }
        for err, cnt in sorted(failure_reasons.items(), key=lambda x: -x[1])
    }

    prefix_stats: Dict[str, dict] = {}
    for r in results:
        iid = r["instance_id"]
        pfx = next((p for p in ("bq", "ga", "local", "sf_bq", "sf") if iid.startswith(p)), "other")
        s = prefix_stats.setdefault(pfx, {"total": 0, "correct": 0, "skipped": 0})
        s["total"] += 1
        if r["score"] == 1:
            s["correct"] += 1
        if r["score"] == -1:
            s["skipped"] += 1

    return {
        "mode": mode,
        "pred_file": pred_file,
        "evaluated_count": total_eval,
        "skipped_count": len(skipped),
        "correct": correct,
        "ex_accuracy": correct / total_eval if total_eval else 0.0,
        "official_accuracy": correct / TOTAL_OFFICIAL,
        "execute_success_count": exec_ok,
        "execute_success_rate": exec_ok / total_eval if total_eval else 0.0,
        "execute_failure_count": exec_fail,
        "execute_failure_rate": exec_fail / total_eval if total_eval else 0.0,
        "failure_reasons": failure_detail,
        "prefix_breakdown": {
            pfx: {
                "total": s["total"],
                "correct": s["correct"],
                "skipped": s["skipped"],
                "accuracy": s["correct"] / s["total"] if s["total"] else 0.0,
            }
            for pfx, s in sorted(prefix_stats.items())
        },
    }


def _save_log(summary: dict, log_dir: str, pred_file: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H")
    name = f"eval_stats_spider2_{Path(pred_file).stem}_{ts}.log"
    path = os.path.join(log_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Spider2-Lite 评测（local SQLite + BigQuery，输出与 BIRD eval 格式一致的 log）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pred", type=str, required=True, help="infer_spider2.py 输出的 JSON")
    parser.add_argument(
        "--mode",
        choices=["local_only", "bq", "official"],
        default="local_only",
        help=(
            "local_only: 只跑 135 条 SQLite，无需凭证；"
            "bq: SQLite + BigQuery（bq/ga），需 bigquery_credential.json；"
            "official: 调用官方 evaluate.py（等价于 bq 模式，sf_bq/* 同样被跳过）"
        ),
    )
    parser.add_argument(
        "--spider2_dir",
        default="/data/banlujie/data/spider2/spider2-lite",
    )
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=60, help="SQLite 执行超时（秒）")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--log_dir",
        default="/data/banlujie/RL-NL2SQL/eval_res/logs",
        help="summary log 目录（与 BIRD eval 日志同目录）",
    )
    args = parser.parse_args()

    spider2_dir = Path(args.spider2_dir)
    eval_suite = spider2_dir / "evaluation_suite"
    gold_dir = eval_suite / "gold"
    gold_result_dir = str(gold_dir / "exec_result")
    eval_jsonl = str(gold_dir / "spider2lite_eval.jsonl")
    db_base = spider2_dir / "resource" / "databases"
    bq_credential = str(eval_suite / "bigquery_credential.json")

    # ── 模式 official：直接调官方脚本 ─────────────────────────────────────────
    if args.mode == "official":
        pred_dir = Path(args.pred).parent
        model_name = Path(args.pred).stem.replace("spider2_infer_", "")
        sql_dir = pred_dir / f"sql_{model_name}"
        if not sql_dir.exists():
            print(f"错误：未找到 SQL 文件目录 {sql_dir}")
            print("请先运行 infer_spider2.py，它会自动写出每条 <instance_id>.sql")
            sys.exit(1)
        print(f"调用官方 evaluate.py")
        print(f"  SQL 目录:   {sql_dir}")
        print(f"  cwd:        {eval_suite}  （bigquery_credential.json 在此目录，相对路径自动生效）")
        print(f"  注意: sf_bq/* sf/* 样本官方也无法执行，会自动跳过")
        ret = subprocess.run(
            [
                sys.executable, str(eval_suite / "evaluate.py"),
                "--mode", "sql",
                "--result_dir", str(sql_dir),
                "--gold_dir", str(gold_dir),
                "--max_workers", str(args.max_workers),
                "--timeout", str(args.timeout),
            ],
            cwd=str(eval_suite),
        )
        sys.exit(ret.returncode)

    # ── 模式 local_only / bq ──────────────────────────────────────────────────
    eval_bq = (args.mode == "bq")
    if eval_bq and not Path(bq_credential).exists():
        print(f"错误：BigQuery 凭证不存在: {bq_credential}")
        sys.exit(1)

    with open(args.pred, "r", encoding="utf-8") as f:
        pred_data: List[dict] = json.load(f)
    print(f"推理结果: {len(pred_data)} 条")

    eval_std: Dict[str, dict] = {}
    with open(eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line.strip())
            eval_std[it["instance_id"]] = it

    gold_ids = set(eval_std.keys())
    eval_items = [d for d in pred_data if d["instance_id"] in gold_ids]

    # 按类型分组打印
    by_engine: Dict[str, int] = {}
    for it in eval_items:
        e = _db_engine(it["instance_id"])
        by_engine[e] = by_engine.get(e, 0) + 1
    print(f"与 gold 匹配: {len(eval_items)} 条  分布: {dict(sorted(by_engine.items()))}")
    if eval_bq:
        print(f"模式 bq: SQLite + BigQuery (bq/ga) 全量执行")
        print(f"  BigQuery 凭证: {bq_credential}")
    else:
        print(f"模式 local_only: 只执行 SQLite，bq/ga/sf 标记为 skipped")

    tmp_path = Path(tempfile.mkdtemp(prefix="spider2_eval_"))
    results = []
    try:
        workers = min(args.max_workers, max(len(eval_items), 1))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    _eval_one, item, eval_std, gold_result_dir, db_base, tmp_path,
                    args.timeout, bq_credential, eval_bq,
                ): item["instance_id"]
                for item in eval_items
            }
            for fut in tqdm(as_completed(futs), total=len(eval_items), desc="Evaluating"):
                results.append(fut.result())
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    results.sort(key=lambda x: x["instance_id"])
    summary = _build_summary(results, args.mode, args.pred)

    # ── 打印汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"Spider2-Lite 评测结果  [{args.mode}]")
    print("=" * 65)
    print(f"  已评测:          {summary['evaluated_count']} / {TOTAL_OFFICIAL} 条")
    print(f"  跳过:            {summary['skipped_count']} 条")
    print(f"  正确:            {summary['correct']} 条")
    print(f"  EX（已评测分母）:{summary['ex_accuracy']:.4f}  ({summary['ex_accuracy']:.2%})")
    print(f"  官方口径 (/547): {summary['official_accuracy']:.4f}  ({summary['official_accuracy']:.2%})")
    print()
    print("  前缀分布:")
    for pfx, s in summary["prefix_breakdown"].items():
        skip_str = f"  skipped={s['skipped']}" if s["skipped"] else ""
        print(f"    {pfx:<8} total={s['total']:3d}  correct={s['correct']:3d}  "
              f"acc={s['accuracy']:.2%}{skip_str}")
    if summary["failure_reasons"]:
        print()
        print("  失败原因 (top 10):")
        for err, info in list(summary["failure_reasons"].items())[:10]:
            print(f"    {info['count']:3d}  {err}")
    print("=" * 65)

    # ── 保存详细结果 ──────────────────────────────────────────────────────────
    out_path = args.output or str(Path(args.pred).parent / f"{Path(args.pred).stem}_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({**summary, "details": results}, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果: {out_path}")

    # ── 写 summary log（与 BIRD eval 同格式）──────────────────────────────────
    log_summary = {k: v for k, v in summary.items()}
    log_path = _save_log(log_summary, args.log_dir, args.pred)
    print(f"Summary log: {log_path}")


if __name__ == "__main__":
    from typing import List
    main()
