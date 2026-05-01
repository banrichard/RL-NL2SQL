import sys
import pysqlite3 as sqlite3
import json
import argparse
import os
from datetime import datetime
import re
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import multiprocessing as mp
import random

random.seed(42)

execution_results = None
evaluation_results = None


def _rewrite_mmddyyyy_to_iso(sql: str) -> str:
    """Rewrite date literals like '9/9/2019' or '9-9-2019' to '2019-09-09'."""
    if not sql:
        return sql

    def _repl(match):
        m = int(match.group(1))
        d = int(match.group(2))
        y = int(match.group(3))
        try:
            datetime(y, m, d)
        except ValueError:
            return match.group(0)
        return f"'{y:04d}-{m:02d}-{d:02d}'"

    out = re.sub(r"'(\d{1,2})/(\d{1,2})/(\d{4})'", _repl, sql)
    out = re.sub(r"'(\d{1,2})-(\d{1,2})-(\d{4})'", _repl, out)
    return out


def _quote_ident(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def _get_table_columns(conn, table_name: str):
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({_quote_ident(table_name)})")
        rows = cur.fetchall()
        return {str(r[1]).lower() for r in rows if len(r) > 1 and r[1] is not None}
    except Exception:
        return set()


def _repair_unknown_alias(sql: str, db_file: str) -> str:
    """
    Repair references like m.college where alias `m` was never defined.
    If all columns of an unknown alias uniquely map to one table used in the query,
    replace `alias.` with `table.`.
    """
    if not sql:
        return sql
    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return sql

    try:
        root = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return sql

    tables = []
    known = set()
    for t in root.find_all(exp.Table):
        table_name = (t.name or "").strip()
        if not table_name:
            continue
        alias_name = (t.alias_or_name or "").strip()
        tables.append(table_name)
        known.add(table_name.lower())
        if alias_name:
            known.add(alias_name.lower())

    if not tables:
        return sql

    unknown_alias_cols = {}
    for col in root.find_all(exp.Column):
        alias = (col.table or "").strip()
        col_name = (col.name or "").strip()
        if not alias or not col_name:
            continue
        if alias.lower() in known:
            continue
        unknown_alias_cols.setdefault(alias.lower(), set()).add(col_name.lower())

    if not unknown_alias_cols:
        return sql

    try:
        conn = sqlite3.connect(db_file)
    except Exception:
        return sql
    try:
        table_cols = {t: _get_table_columns(conn, t) for t in tables}
    finally:
        conn.close()

    alias_to_table = {}
    for alias, cols in unknown_alias_cols.items():
        candidates = []
        for t in tables:
            if cols.issubset(table_cols.get(t, set())):
                candidates.append(t)
        if len(candidates) != 1:
            return sql
        alias_to_table[alias] = candidates[0]

    repaired = sql
    for alias, table_name in alias_to_table.items():
        repaired = re.sub(
            rf"(?<![A-Za-z0-9_`]){re.escape(alias)}\s*\.",
            f"{table_name}.",
            repaired,
            flags=re.IGNORECASE,
        )
    return repaired


def _normalize_symbol_literal_spacing(sql: str) -> str:
    """
    Normalize literals like ' = ' / ' >= ' to '=' / '>='.
    This fixes common generation artifacts in symbolic categorical columns.
    """
    if not sql:
        return sql
    return re.sub(r"'\s*([=!<>]{1,2})\s*'", r"'\1'", sql)


def _rewrite_case_insensitive_eq(sql: str) -> str:
    """
    Rewrite col = 'token' as LOWER(col) = LOWER('token') for simple alphabetic tokens.
    Keep it conservative to avoid broad semantic drift.
    """
    if not sql:
        return sql
    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return sql

    try:
        root = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return sql

    changed = False

    def _transform(node):
        nonlocal changed
        if not isinstance(node, exp.EQ):
            return node
        left = node.this
        right = node.expression
        pairs = [(left, right), (right, left)]
        for col_node, lit_node in pairs:
            if isinstance(col_node, exp.Column) and isinstance(lit_node, exp.Literal) and lit_node.is_string:
                token = (lit_node.this or "").strip()
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", token):
                    changed = True
                    return exp.EQ(
                        this=exp.Lower(this=col_node.copy()),
                        expression=exp.Lower(this=lit_node.copy()),
                    )
        return node

    rewritten = root.transform(_transform).sql(dialect="sqlite")
    return rewritten if changed else sql


def _sql_candidates(sql: str, db_file: str):
    candidates = []
    if sql is None:
        return candidates
    steps = [sql]
    s1 = _normalize_symbol_literal_spacing(sql)
    if s1 != sql:
        steps.append(s1)
    d1 = _rewrite_mmddyyyy_to_iso(sql)
    if d1 != sql:
        steps.append(d1)
    d2 = _rewrite_mmddyyyy_to_iso(s1)
    if d2 != s1:
        steps.append(d2)

    # conservative case-insensitive equality rewrite
    for s in list(steps):
        c = _rewrite_case_insensitive_eq(s)
        if c != s:
            steps.append(c)

    # apply alias repair on both original and date-normalized candidates
    for s in list(steps):
        r = _repair_unknown_alias(s, db_file)
        if r != s:
            steps.append(r)

    # keep order, deduplicate
    seen = set()
    for s in steps:
        if s not in seen:
            seen.add(s)
            candidates.append(s)
    return candidates


def _format_pred_error(e: Exception) -> str:
    failure_reason = f"pred_sql_error:{type(e).__name__}"
    if isinstance(e, sqlite3.OperationalError):
        msg = str(e).lower()
        if "incomplete input" in msg:
            failure_reason = "pred_sql_error:OperationalError:incomplete_input"
        elif "no such column" in msg:
            failure_reason = "pred_sql_error:OperationalError:no_such_column"
        elif "syntax error" in msg:
            failure_reason = "pred_sql_error:OperationalError:syntax_error"
        elif "unrecognized token" in msg:
            failure_reason = "pred_sql_error:OperationalError:unrecognized_token"
        elif "misuse of aggregate" in msg:
            failure_reason = "pred_sql_error:OperationalError:aggregate_misuse"
    return failure_reason


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type = str, default = "predict_dev.json")
    parser.add_argument('--gold', type = str, default = "/data/banlujie/data/bird_dev/dev.json")
    parser.add_argument('--db_path', type = str, default = "/data/banlujie/data/bird_dev/dev_databases")
    parser.add_argument('--mode', type = str, default = "greedy_search")
    parser.add_argument('--num_cpus', type=int, default=20)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--log_dir', type=str, default="/data/banlujie/RL-NL2SQL/eval_res/logs")
    parser.add_argument('--disable_post_process', action='store_true', help="Disable SQL post-processing during evaluation.")

    opt = parser.parse_args()

    return opt

def execute_sql(data_idx, db_file, sql, enable_post_process=True):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    last_exception = None
    try:
        conn.execute("BEGIN TRANSACTION;")
        best_sql = None
        best_res = None
        if enable_post_process:
            candidate_sqls = _sql_candidates(sql, db_file)
        else:
            candidate_sqls = [sql]
        for cand_sql in candidate_sqls:
            try:
                cursor.execute(cand_sql)
                execution_res = cursor.fetchall()
                execution_res = frozenset(execution_res) # make set hashable
                if best_res is None or (len(best_res) == 0 and len(execution_res) > 0):
                    best_sql = cand_sql
                    best_res = execution_res
            except Exception as e:
                last_exception = e
        if best_res is not None:
            conn.rollback()
            conn.close()
            return data_idx, db_file, best_sql, best_res, 1

        # if len(execution_res) > 0:
        #     return data_idx, db_file, sql, execution_res, 1
        # elif len(execution_res) == 0:
        #     return data_idx, db_file, sql, execution_res, 0
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, None, 0
    except Exception:
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, None, 0

def compare_sql(question_id, db_file, question, ground_truth, pred_sql, enable_post_process=True) :
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    correctness = 0
    execute_success = 0  # 0: SQL执行失败, 1: SQL执行成功
    failure_reason = ""

    if pred_sql is None or str(pred_sql).strip() == "":
        conn.close()
        return question_id, db_file, question, ground_truth, pred_sql, correctness, execute_success, "empty_prediction"
    if str(pred_sql).strip().lower() == "error sql":
        conn.close()
        return question_id, db_file, question, ground_truth, pred_sql, correctness, execute_success, "invalid_prediction"

    try:
        conn.execute("BEGIN TRANSACTION;")
        try:
            cursor.execute(ground_truth)
            ground_truth_res = cursor.fetchall()
        except Exception as e:
            failure_reason = f"ground_truth_sql_error:{type(e).__name__}"
            conn.rollback()
            return question_id, db_file, question, ground_truth, pred_sql, correctness, execute_success, failure_reason

        last_exception = None
        saw_success = False
        if enable_post_process:
            candidate_sqls = _sql_candidates(pred_sql, db_file)
        else:
            candidate_sqls = [pred_sql]
        for cand_sql in candidate_sqls:
            try:
                cursor.execute(cand_sql)
                predicted_res = cursor.fetchall()
                saw_success = True
                if set(predicted_res) == set(ground_truth_res):
                    correctness = 1
                    failure_reason = "correct"
                    execute_success = 1
                    conn.rollback()
                    return question_id, db_file, question, ground_truth, cand_sql, correctness, execute_success, failure_reason
            except Exception as e:
                last_exception = e

        if saw_success:
            execute_success = 1
            failure_reason = "execution_mismatch"
        else:
            execute_success = 0
            failure_reason = _format_pred_error(last_exception) if last_exception else "pred_sql_error:unknown"
        conn.rollback()
    finally:
        conn.close()
    return question_id, db_file, question, ground_truth, pred_sql, correctness, execute_success, failure_reason

def compare_sql_wrapper(args, timeout):
    '''Wrap execute_sql for timeout'''
    question_id, db_file, question, ground_truth, pred_sql, *_ = args
    try:
        result = func_timeout(timeout, compare_sql, args=args)
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = (question_id, db_file, question, ground_truth, pred_sql, 0, 0, "timeout")
    except Exception as e:
        result = (
            question_id,
            db_file,
            question,
            ground_truth,
            pred_sql,
            0,
            0,
            f"wrapper_error:{type(e).__name__}",
        )
    return result

def execute_sql_wrapper(data_idx, db_file, sql, timeout, enable_post_process=True):
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql, enable_post_process))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        print(f"Data index:{data_idx}\nSQL:\n{sql}\nTime Out!")
        print("-"*30)
        res = (data_idx, db_file, sql, None, 0)
    except Exception as e:
        res = (data_idx, db_file, sql, None, 0)

    return res

def execute_callback_evaluate_sql(result):
    '''Store the execution result in the collection'''
    question_id, db_file, question, ground_truth, pred_sql, correctness, execute_success, failure_reason = result
    # evaluation_res = dict()
    # evaluation_res['question_id'] = question_id
    # evaluation_res["db_file"] = db_file
    # evaluation_res["question"] = question
    # evaluation_res["ground_truth"] = ground_truth
    # evaluation_res["pred_sql"] = pred_sql
    # evaluation_res["correctness"] = correctness
    evaluation_results.append(
        {
            "question_id": question_id,
            "db_file": db_file,
            "question": question,
            "ground_truth": ground_truth,
            "pred_sql": pred_sql,
            "correctness": correctness,
            "execute_success": execute_success,
            "failure_reason": failure_reason
        }
    )

    print('Done:', question_id, 'correctness:', correctness, 'execute_success:', execute_success, 'reason:', failure_reason) # Print the progress
    sys.stdout.flush()
    sys.stderr.flush()

def summarize_eval_results(evaluation_results, mode):
    total = len(evaluation_results)
    correctness_scores = [res["correctness"] for res in evaluation_results]
    ex_accuracy = (sum(correctness_scores) / total) if total else 0.0

    execute_success_scores = [res["execute_success"] for res in evaluation_results]
    execute_success_count = sum(execute_success_scores)
    execute_success_rate = (execute_success_count / total) if total else 0.0

    failure_counter = {}
    failure_count = 0
    for res in evaluation_results:
        if res["execute_success"] == 0:
            failure_count += 1
            reason = res.get("failure_reason", "") or "unknown"
            failure_counter[reason] = failure_counter.get(reason, 0) + 1

    failure_reason_ratio = {}
    for reason, count in sorted(failure_counter.items(), key=lambda x: (-x[1], x[0])):
        failure_reason_ratio[reason] = {
            "count": count,
            "ratio_in_failures": (count / failure_count) if failure_count else 0.0,
            "ratio_in_all_samples": (count / total) if total else 0.0,
        }

    return {
        "mode": mode,
        "sample_count": total,
        "ex_accuracy": ex_accuracy,
        "acc": ex_accuracy,
        "execute_success_count": execute_success_count,
        "execute_success_rate": execute_success_rate,
        "execute_failure_count": failure_count,
        "execute_failure_rate": (failure_count / total) if total else 0.0,
        "failure_reasons": failure_reason_ratio,
    }

def write_eval_log(stats, pred_file, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp_hour = datetime.now().strftime("%Y%m%d_%H")
    pred_name = os.path.splitext(os.path.basename(pred_file))[0]
    log_path = os.path.join(log_dir, f"eval_stats_{pred_name}_{timestamp_hour}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Evaluation stats log saved to: {log_path}")
    return log_path

def execute_callback_execute_sqls(result):
    data_idx, db_file, sql, query_result, valid = result
    # print('Done:', data_idx) # Print the progress

    execution_results.append(
        {
            "data_idx": data_idx,
            "db_file": db_file,
            "sql": sql,
            "query_result": query_result,
            "valid": valid
        }
    )

def evaluate_sqls_parallel(db_files, questions, pred_sqls, ground_truth_sqls, num_cpus=1, timeout=1, enable_post_process=True):
    '''Execute the sqls in parallel'''
    pool = mp.Pool(processes=num_cpus)
    for question_id, db_file, question, pred_sql, ground_truth in zip([x for x in range(len(db_files))], db_files, questions, pred_sqls, ground_truth_sqls):
        pool.apply_async(
            compare_sql_wrapper,
            args=((question_id, db_file, question, ground_truth, pred_sql, enable_post_process), timeout),
            callback=execute_callback_evaluate_sql,
        )
    pool.close()
    pool.join()

def execute_sqls_parallel(db_files, sqls, num_cpus=1, timeout=1, enable_post_process=True):
    pool = mp.Pool(processes=num_cpus)
    for data_idx, db_file, sql in zip(list(range(len(sqls))), db_files, sqls):
        pool.apply_async(
            execute_sql_wrapper,
            args=(data_idx, db_file, sql, timeout, enable_post_process),
            callback=execute_callback_execute_sqls,
        )
    pool.close()
    pool.join()

def mark_invalid_sqls(db_files, sqls, num_cpus=20, timeout=10, enable_post_process=True):
    global execution_results
    execution_results = []
    execute_sqls_parallel(db_files, sqls, num_cpus=num_cpus, timeout=timeout, enable_post_process=enable_post_process)
    execution_results = sorted(execution_results, key=lambda x:x['data_idx'])
    
    for idx, res in enumerate(execution_results):
        if res["valid"] == 0:
            sqls[idx] = "Error SQL"
    return sqls

def major_voting(db_files, pred_sqls, sampling_num, return_random_one_when_all_errors=True, num_cpus=20, timeout=10, enable_post_process=True):
    global execution_results
    mj_pred_sqls = []
    execution_results = []
    # execute all sampled SQL queries to obtain their execution results
    execute_sqls_parallel(db_files, pred_sqls, num_cpus=num_cpus, timeout=timeout, enable_post_process=enable_post_process)
    execution_results = sorted(execution_results, key=lambda x:x['data_idx'])
    print("len(execution_results):", len(execution_results))

    # perform major voting
    for result_idx in range(0, len(execution_results), sampling_num):
        major_voting_counting = dict()
        execution_results_of_one_sample = execution_results[result_idx: result_idx + sampling_num]

        # if no predicted SQLs are valid
        if sum([res["valid"] for res in execution_results_of_one_sample]) == 0:
            if return_random_one_when_all_errors:
                mj_pred_sql = random.choice(execution_results_of_one_sample)["sql"] # select a random one to return
            else:
                mj_pred_sql = "Error SQL"
            mj_pred_sqls.append(mj_pred_sql)
            continue

        for res in execution_results_of_one_sample:
            if res["valid"] == 1: # skip invalid SQLs
                if res["query_result"] in major_voting_counting:
                    major_voting_counting[res["query_result"]]["votes"] += 1
                else:
                    major_voting_counting[res["query_result"]] = {"votes": 1, "sql": res["sql"]}
        
        # find the SQL with the max votes
        major_vote = max(major_voting_counting.values(), key=lambda x: x["votes"])
        mj_pred_sql = major_vote["sql"]
        mj_pred_sqls.append(mj_pred_sql)
    
    return mj_pred_sqls

def run_eval(gold_file, pred_file, db_path, mode, save_pred_sqls, num_cpus=20, timeout=10, log_dir="/data/banlujie/RL-NL2SQL/eval_res/logs", enable_post_process=True):
    global evaluation_results
    gold = json.load(open(gold_file))
    if not os.path.exists(pred_file):
        pred_dir = os.path.dirname(pred_file) or "."
        pred_basename = os.path.basename(pred_file)
        similar_files = []
        if os.path.isdir(pred_dir):
            name_prefix = pred_basename.split(".")[0]
            similar_files = sorted(
                [f for f in os.listdir(pred_dir) if f.endswith(".json") and name_prefix in f]
            )
        hint = f"No such file: {pred_file}"
        if similar_files:
            hint += f"\nPossible candidates in {pred_dir}: {similar_files[:5]}"
        hint += "\nTip: infer.py saves voting results as <output_name>_<model_name>.json"
        raise FileNotFoundError(hint)
    pred_results = json.load(open(pred_file))
    db_files = [os.path.join(db_path, data["db_id"], data["db_id"] + ".sqlite") for data in gold]
    questions = [data["question"] for data in gold]
    pred_sql_key = "pred_sqls"
    # pred_sql_key = "responses"

    if "bird" in gold_file:
        ground_truth_sqls = [data["SQL"] for data in gold]
    else:
        ground_truth_sqls = [data["query"] for data in gold]

    if mode == "greedy_search":
        pred_sqls = [res[pred_sql_key][0] for res in pred_results]

        # save the (greedy-search) predicted SQL so we can check it out later
        if save_pred_sqls:
            with open(pred_file[:-5] + "_pred_greedy_search_sqls.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(pred_sqls, indent=2 ,ensure_ascii=False))
        
        assert len(pred_results) == len(pred_sqls) == len(db_files) == len(questions) == len(ground_truth_sqls)

        evaluation_results = []
        evaluate_sqls_parallel(
            db_files,
            questions,
            pred_sqls,
            ground_truth_sqls,
            num_cpus=num_cpus,
            timeout=timeout,
            enable_post_process=enable_post_process,
        )

        # sort evaluation_results by question_id
        evaluation_results = sorted(evaluation_results, key=lambda x:x['question_id'])
        evaluation_scores = [res["correctness"] for res in evaluation_results]
        for res in evaluation_results:
            if res["correctness"] == 0:
                print("question:", res["question"])
                print("GT:", res["ground_truth"])
                print("Pred:", res["pred_sql"])
                print("-"*30)
        stats = summarize_eval_results(evaluation_results, mode)
        print("EX Accuracy (greedy search):", stats["ex_accuracy"])
        print("Execute Success Count:", stats["execute_success_count"])
        print("Execute Success Rate:", stats["execute_success_rate"])
        print("Execute Failure Count:", stats["execute_failure_count"])
        print("Execute Failure Reasons:", json.dumps(stats["failure_reasons"], ensure_ascii=False))
        write_eval_log(stats, pred_file, log_dir)

        return stats["ex_accuracy"], pred_sqls
    elif mode == "major_voting":
        sampling_num = len(pred_results[0][pred_sql_key])
        print("sampling_num:", sampling_num)

        db_files = []
        for gold_data in gold:
            db_files.extend([os.path.join(db_path, gold_data["db_id"], gold_data["db_id"] + ".sqlite")] * sampling_num)

        pred_sqls = []
        for pred_data in pred_results:
            pred_sqls.extend(pred_data[pred_sql_key])
        assert len(pred_sqls) == len(db_files)

        mj_pred_sqls = major_voting(
            db_files,
            pred_sqls,
            sampling_num,
            num_cpus=num_cpus,
            timeout=timeout,
            enable_post_process=enable_post_process,
        )

        # save the (major-voting) predicted SQL so we can check it out later
        if save_pred_sqls:
            with open(pred_file[:-5] + "_pred_major_voting_sqls.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(mj_pred_sqls, indent=2 ,ensure_ascii=False))

        # reset db_files
        db_files = []
        for gold_data in gold:
            db_files.append(os.path.join(db_path, gold_data["db_id"], gold_data["db_id"] + ".sqlite"))

        assert len(mj_pred_sqls) == len(db_files) == len(questions) == len(ground_truth_sqls)

        evaluation_results = []
        evaluate_sqls_parallel(
            db_files,
            questions,
            mj_pred_sqls,
            ground_truth_sqls,
            num_cpus=num_cpus,
            timeout=timeout,
            enable_post_process=enable_post_process,
        )

        # sort evaluation_results by question_id
        evaluation_results = sorted(evaluation_results, key=lambda x:x['question_id'])
        stats = summarize_eval_results(evaluation_results, mode)
        print("EX Accuracy (major voting):", stats["ex_accuracy"])
        print("Execute Success Count:", stats["execute_success_count"])
        print("Execute Success Rate:", stats["execute_success_rate"])
        print("Execute Failure Count:", stats["execute_failure_count"])
        print("Execute Failure Reasons:", json.dumps(stats["failure_reasons"], ensure_ascii=False))
        write_eval_log(stats, pred_file, log_dir)

        return stats["ex_accuracy"], mj_pred_sqls
    elif mode == "pass@k":
        all_scores = []
        all_execute_success = []  # 保存所有采样的execute_success分数
        sampling_num = len(pred_results[0][pred_sql_key])

        db_files = []
        for gold_data in gold:
            db_files.append(os.path.join(db_path, gold_data["db_id"], gold_data["db_id"] + ".sqlite"))

        for sample_idx in range(sampling_num):
            pred_sqls_for_specific_sample_idx = [pred_data[pred_sql_key][sample_idx] for pred_data in pred_results]
            evaluation_results = []
            evaluate_sqls_parallel(
                db_files,
                questions,
                pred_sqls_for_specific_sample_idx,
                ground_truth_sqls,
                num_cpus=num_cpus,
                timeout=timeout,
                enable_post_process=enable_post_process,
            )
            evaluation_results = sorted(evaluation_results, key=lambda x:x['question_id'])
            evaluation_scores = [res["correctness"] for res in evaluation_results]
            execute_success_scores = [res["execute_success"] for res in evaluation_results]
            all_scores.append(evaluation_scores)
            all_execute_success.append(execute_success_scores)
        
        pass_at_k_scores = [1 if any(column) else 0 for column in zip(*all_scores)]
        ex_accuracy = sum(pass_at_k_scores)/len(pass_at_k_scores)
        print(f"EX Accuracy (pass@{sampling_num}):", ex_accuracy)
        
        # 统计SQL执行成功的数量（对所有采样）
        execute_success_at_k_scores = [1 if any(column) else 0 for column in zip(*all_execute_success)]
        execute_success_count = sum(execute_success_at_k_scores)
        execute_success_rate = execute_success_count/len(execute_success_at_k_scores)
        print("Execute Success Count (pass@k):", execute_success_count)
        print("Execute Success Rate (pass@k):", execute_success_rate)

        stats = {
            "mode": mode,
            "sample_count": len(pass_at_k_scores),
            "sampling_num": sampling_num,
            "ex_accuracy": ex_accuracy,
            "acc": ex_accuracy,
            "execute_success_count": execute_success_count,
            "execute_success_rate": execute_success_rate,
            "execute_failure_count": len(pass_at_k_scores) - execute_success_count,
            "execute_failure_rate": 1 - execute_success_rate,
            "failure_reasons": {
                "pass_at_k_does_not_track_single_failure_reason": {
                    "count": len(pass_at_k_scores) - execute_success_count,
                    "ratio_in_failures": 1.0 if execute_success_count != len(pass_at_k_scores) else 0.0,
                    "ratio_in_all_samples": (len(pass_at_k_scores) - execute_success_count) / len(pass_at_k_scores) if pass_at_k_scores else 0.0,
                }
            },
        }
        write_eval_log(stats, pred_file, log_dir)
        
        return ex_accuracy, None
    else:
        raise ValueError("mode should be in [greedy_search, major_voting, pass@k]")

if __name__ == "__main__":
    opt = parse_option()
    run_eval(
        opt.gold,
        opt.pred,
        opt.db_path,
        opt.mode,
        False,
        num_cpus=opt.num_cpus,
        timeout=opt.timeout,
        log_dir=opt.log_dir,
        enable_post_process=not opt.disable_post_process,
    )
