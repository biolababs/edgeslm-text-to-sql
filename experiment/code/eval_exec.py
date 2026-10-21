#!/usr/bin/env python3
"""Score predicted SQL by execution accuracy against the gold answers.

Predictions: JSONL with {"id": ..., "sql": ...}. A missing id counts as wrong.
Markdown code fences around the SQL are stripped.

Two scores are reported:
    strict   the result must match the gold rows exactly (after normalisation)
    lenient  extra output columns are allowed if some subset of columns matches,
             e.g. returning (zone, count) when the gold query returns zone only

Normalisation: numbers compare with tolerance (abs 0.01 or rel 1e-4), numeric
strings count as numbers, string comparison ignores case and surrounding
whitespace. Row order matters only when the example has "ordered": true.

Usage:
    python eval_exec.py --db data/taxi.db --pred preds.jsonl \
        --gold data/questions/test_seen.jsonl data/questions/test_unseen.jsonl
    python eval_exec.py --db data/taxi.db --oracle --gold data/questions/*.jsonl
"""
import argparse
import itertools
import json
import math
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

MAX_ROWS = 10_000
MAX_PROJECTED_COLS = 8
FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)


def norm(v):
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s)
        except ValueError:
            return s.casefold()
    if isinstance(v, (int, float)):
        return float(v)
    return v


def sort_key(row):
    # None < numbers < strings; numbers rounded so near-equal values sort together
    return tuple((0, 0) if v is None else (1, round(v, 2)) if isinstance(v, float) else (2, str(v))
                 for v in row)


def same_value(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-4, abs_tol=0.01)
    return a == b


def same_rows(pred, gold, ordered):
    if len(pred) != len(gold):
        return False
    if not ordered:
        pred, gold = sorted(pred, key=sort_key), sorted(gold, key=sort_key)
    return all(len(p) == len(g) and all(same_value(x, y) for x, y in zip(p, g))
               for p, g in zip(pred, gold))


def lenient_match(pred, gold, ordered):
    if not gold or not pred:
        return same_rows(pred, gold, ordered)
    n_gold, n_pred = len(gold[0]), len(pred[0])
    if n_pred < n_gold or n_pred > MAX_PROJECTED_COLS:
        return False
    for cols in itertools.permutations(range(n_pred), n_gold):
        if same_rows([[r[c] for c in cols] for r in pred], gold, ordered):
            return True
    return False


def execute(conn, sql, timeout_s):
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    try:
        rows = conn.execute(sql).fetchmany(MAX_ROWS + 1)
    except (sqlite3.Error, sqlite3.Warning) as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        conn.set_progress_handler(None, 0)
    if len(rows) > MAX_ROWS:
        return None, "too many rows"
    return [[norm(v) for v in r] for r in rows], None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("data/taxi.db"))
    ap.add_argument("--gold", type=Path, nargs="+", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred", type=Path, help="predictions JSONL")
    src.add_argument("--oracle", action="store_true", help="score the gold SQL itself (sanity check)")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds per query")
    ap.add_argument("--by-template", action="store_true", help="also break scores down by template")
    ap.add_argument("--details", type=Path, help="write per-example results to this JSONL")
    args = ap.parse_args()

    gold = []
    for path in args.gold:
        with open(path, encoding="utf-8") as f:
            gold += [json.loads(line) for line in f if line.strip()]

    preds = {}
    if args.pred:
        with open(args.pred, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    preds[p["id"]] = p.get("sql") or ""

    # read-only: model SQL cannot modify the database
    conn = sqlite3.connect(f"{args.db.resolve().as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA mmap_size = 1073741824")

    totals = defaultdict(lambda: [0, 0, 0])  # key -> [n, strict, lenient]
    details = []
    for ex in gold:
        sql = ex["sql"] if args.oracle else FENCE.sub("", preds.get(ex["id"], "")).strip()
        gold_rows = [[norm(v) for v in r] for r in ex["answer"]]
        if sql:
            rows, err = execute(conn, sql, args.timeout)
        else:
            rows, err = None, "no prediction"
        strict = rows is not None and same_rows(rows, gold_rows, ex["ordered"])
        lenient = strict or (rows is not None and lenient_match(rows, gold_rows, ex["ordered"]))
        keys = [ex["split"]]
        if args.by_template and ex.get("template_id"):
            keys.append(f"{ex['split']} / {ex['template_id']}")
        for k in keys:
            totals[k][0] += 1
            totals[k][1] += strict
            totals[k][2] += lenient
        details.append({"id": ex["id"], "split": ex["split"], "strict": strict,
                        "lenient": lenient, "error": err, "pred_sql": sql})
    conn.close()

    width = max(len(k) for k in totals) + 2
    print(f"{'split':<{width}}{'n':>6}{'strict':>9}{'lenient':>9}")
    for k in sorted(totals, key=lambda k: (" / " in k, k)):
        n, s, l = totals[k]
        print(f"{k:<{width}}{n:>6}{100 * s / n:>8.1f}%{100 * l / n:>8.1f}%")

    if args.details:
        with open(args.details, "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
