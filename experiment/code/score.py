#!/usr/bin/env python3
"""Score a run folder on the PC (PLAN 6.1, D13). Writes <run>/scores/ (derived, re-creatable).

For every <run>/acc/<model>/<cond>.jsonl:
  NYC conditions (nyc0, nycsim, nycfix, nycnat, cur*, lora_*): strict and lenient EX with
      eval_exec.py's executor/normalisation against gold answers computed on the PC; error taxonomy.
  spider0: official test-suite-sql-eval evaluation.py on the original dev DBs (EX), on the
      distilled test-suite DBs (TS) and exact-set match (EM), on the subset of items present.
  bird0 / birdnat: official mini_dev evaluation_ex.py and evaluation_f1.py (SQLite, 30 s timeout);
      plus the corrected Mini-Dev (Arcwise-Plat-SQL) re-score on the items whose question and
      evidence are unchanged from v1.
Also: per-query energy from the session power log(s) in <run>/power/ (trapezoid over
[t_send, t_recv], total and above the run's idle baseline), and --oracle mode.

    python score.py --run results/raw/pilot-20261016 --cfg experiment/config/score_paths.json
    python score.py --oracle --cfg experiment/config/score_paths.json --out build/oracle
"""
import argparse
import json
import re
import sqlite3
import statistics as st
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_exec import execute, lenient_match, norm, same_rows  # noqa: E402
from prompts import extract_sql  # noqa: E402

VDD = re.compile(r"VDD_IN (\d+)mW")


def rd(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def gold_nyc(cfg):
    g = {}
    for name in ("nyc_eval", "nyc_pool"):
        for e in rd(Path(cfg["experiment"]) / "data" / f"{name}.jsonl"):
            g[e["id"]] = e
    for e in rd(Path(cfg["experiment"]) / "curated" / "curated.jsonl"):
        g[e["id"]] = e
    return g


def taxonomy(rec, rows, err, strict):
    if strict:
        return "correct"
    if not rec.get("sql"):
        return "truncated" if rec.get("truncated") else "no_sql"
    if rec.get("truncated"):
        return "truncated"
    if err:
        return "exec_error"
    return "wrong_result"


def score_nyc(cfg, recs, gold, conn):
    out = []
    for r in recs:
        g = gold[r["id"]]
        gold_rows = [[norm(v) for v in row] for row in g["answer"]]
        rows, err = execute(conn, r["sql"], 10.0) if r.get("sql") else (None, "no prediction")
        strict = rows is not None and same_rows(rows, gold_rows, g["ordered"])
        lenient = strict or (rows is not None and lenient_match(rows, gold_rows, g["ordered"]))
        out.append({"rid": r["rid"], "id": r["id"], "cond": r["cond"], "strict": strict, "lenient": lenient,
                    "error": err, "class": taxonomy(r, rows, err, strict),
                    # Exposure is carried in the request id tail ("<item>|<fold>|<seen|unseen>");
                    # raw records do not have a separate role field.
                    "role": r.get("role") or (r["rid"].rsplit("|", 1)[-1]
                                              if "|" in r.get("rid", "") else None),
                    "template_id": g.get("template_id"), "fold": g.get("fold")})
    return out


def run_spider(cfg, recs, work, oracle=False):
    """Official evaluator on the subset present, in dataset order."""
    S = Path(cfg["spider_root"]); work.mkdir(parents=True, exist_ok=True)
    dev = json.loads((S / "dev.json").read_text(encoding="utf-8"))
    preds = {r["id"]: " ".join((r.get("sql") or "").split()) or "SELECT 1" for r in recs}
    ids = [f"spider_dev_{i:04d}" for i in range(len(dev)) if f"spider_dev_{i:04d}" in preds]
    gold_lines = [f"{dev[int(i[-4:])]['query']}\t{dev[int(i[-4:])]['db_id']}" for i in ids]
    (work / "gold.sql").write_text("\n".join(gold_lines) + "\n", encoding="utf-8")
    (work / "pred.txt").write_text("\n".join(preds[i] for i in ids) + "\n", encoding="utf-8")
    res = {"n": len(ids)}
    for name, db, et in (("ex", S / "database", "exec"), ("ts", Path(cfg["testsuite_db"]), "exec"),
                         ("em", S / "database", "match")):
        p = subprocess.run([cfg["eval_python"], str(HERE / "run_ts_eval.py"), "--gold", str(work / "gold.sql"), "--pred",
                            str(work / "pred.txt"), "--db", str(db), "--table", str(S / "tables.json"), "--etype", et],
                           cwd=cfg["spider_eval_dir"], capture_output=True, text=True,
                           env={**__import__("os").environ, "NLTK_DATA": cfg["nltk_data"]})
        (work / f"{name}.txt").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
        m = re.search(r"^(execution|exact match)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", p.stdout, re.M)
        res[name] = float(m.group(6)) if m else None
        res[f"{name}_by_hardness"] = [float(x) for x in m.groups()[1:5]] if m else None
    return res


def run_bird(cfg, recs, work, gold_src=None, keep=None, tag="v1"):
    B = Path(cfg["bird_root"]); work.mkdir(parents=True, exist_ok=True)
    v1 = json.loads((B / "mini_dev_sqlite.json").read_text(encoding="utf-8"))
    alt = {str(g["question_id"]): g for g in json.loads(Path(gold_src).read_text(encoding="utf-8"))} if gold_src else None
    preds = {r["id"]: " ".join((r.get("sql") or "").split()) or "SELECT 1" for r in recs}
    items = [(f"bird_minidev_{i:03d}", e) for i, e in enumerate(v1) if f"bird_minidev_{i:03d}" in preds]
    if keep is not None:
        items = [(i, e) for i, e in items if str(e["question_id"]) in keep]
    if not items:
        return {"n": 0}
    predj = {str(k): f"{preds[i]}\t----- bird -----\t{e['db_id']}" for k, (i, e) in enumerate(items)}
    (work / f"pred_{tag}.json").write_text(json.dumps(predj, indent=1), encoding="utf-8")
    gl = [" ".join((alt[str(e["question_id"])]["SQL"] if alt else e["SQL"]).split()) + f"\t{e['db_id']}" for _, e in items]
    (work / f"gold_{tag}.sql").write_text("\n".join(gl) + "\n", encoding="utf-8")
    (work / f"meta_{tag}.jsonl").write_text("".join(json.dumps(e) + "\n" for _, e in items), encoding="utf-8")
    res = {"n": len(items), "ids": [i for i, _ in items]}
    for metric in ("ex", "f1"):
        cmd = [cfg["eval_python"], "-u", f"evaluation_{metric}.py", "--db_root_path", str(B / "dev_databases") + "/",
               "--predicted_sql_path", str(work / f"pred_{tag}.json"), "--ground_truth_path", str(work / f"gold_{tag}.sql"),
               "--num_cpus", "4", "--meta_time_out", "30", "--diff_json_path", str(work / f"meta_{tag}.jsonl"),
               "--sql_dialect", "SQLite", "--output_log_path", str(work / f"{metric}_{tag}.log")]
        env = {**__import__("os").environ, "PYTHONPATH": str(HERE / "sqlite_deadline"), "C9_SQLITE_DEADLINE_S": "30"}
        p = subprocess.run(cmd, cwd=cfg["bird_eval_dir"], capture_output=True, text=True, env=env)
        (work / f"{metric}_{tag}.txt").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
        nums = re.findall(r"^(?:EX|F1|Soft-F1|soft f1|ex)?\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", p.stdout, re.M | re.I)
        res[metric] = float(nums[-1][3]) if nums else None
        res[f"{metric}_raw_tail"] = p.stdout.strip().splitlines()[-4:] if p.stdout.strip() else p.stderr[-500:]
        if res[metric] is None and "ZeroDivisionError" in (p.stdout + p.stderr):
            # The official compute_*_by_diff divides by the count of each difficulty and crashes on
            # subsets lacking one (e.g. the 10-item pilot sample). Fall back to the evaluator's own
            # per-item functions (evaluation_utils.execute_sql + calculate_ex / calculate_f1_score),
            # same 30 s deadline; overall score = mean over items x 100, as the evaluator reports.
            res[metric] = bird_per_item(cfg, work, tag, metric)
            res[f"{metric}_method"] = "per-item official functions (subset lacks a difficulty level)"
    return res


def run_bird_items(cfg, items, work, tag):
    """items: list of dicts {pred, gold, db_id, difficulty, question_id}; official evaluator (+subset fallback)."""
    work.mkdir(parents=True, exist_ok=True)
    if not items:
        return {"n": 0}
    TAB, NL = "\t", "\n"
    predj = {str(k): " ".join(it["pred"].split()) + TAB + "----- bird -----" + TAB + it["db_id"] if it["pred"].strip()
             else "SELECT 1" + TAB + "----- bird -----" + TAB + it["db_id"] for k, it in enumerate(items)}
    (work / f"pred_{tag}.json").write_text(json.dumps(predj, indent=1), encoding="utf-8")
    (work / f"gold_{tag}.sql").write_text(NL.join(" ".join(it["gold"].split()) + TAB + it["db_id"] for it in items) + NL, encoding="utf-8")
    (work / f"meta_{tag}.jsonl").write_text("".join(json.dumps({"question_id": it["question_id"], "db_id": it["db_id"],
                                                              "difficulty": it["difficulty"]}) + NL for it in items), encoding="utf-8")
    B = Path(cfg["bird_root"]); res = {"n": len(items)}
    for metric in ("ex", "f1"):
        cmd = [cfg["eval_python"], "-u", f"evaluation_{metric}.py", "--db_root_path", str(B / "dev_databases") + "/",
               "--predicted_sql_path", str(work / f"pred_{tag}.json"), "--ground_truth_path", str(work / f"gold_{tag}.sql"),
               "--num_cpus", "4", "--meta_time_out", "30", "--diff_json_path", str(work / f"meta_{tag}.jsonl"),
               "--sql_dialect", "SQLite", "--output_log_path", str(work / f"{metric}_{tag}.log")]
        env = {**__import__("os").environ, "PYTHONPATH": str(HERE / "sqlite_deadline"), "C9_SQLITE_DEADLINE_S": "30"}
        p = subprocess.run(cmd, cwd=cfg["bird_eval_dir"], capture_output=True, text=True, env=env)
        (work / f"{metric}_{tag}.txt").write_text(p.stdout + NL + p.stderr, encoding="utf-8")
        nums = re.findall(r"^(?:EX|F1|Soft-F1|soft f1|ex)?\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", p.stdout, re.M | re.I)
        res[metric] = float(nums[-1][3]) if nums else None
        if res[metric] is None and "ZeroDivisionError" in (p.stdout + p.stderr):
            res[metric] = bird_per_item(cfg, work, tag, metric)
            res[f"{metric}_method"] = "per-item official functions (subset lacks a difficulty level)"
    return res


def bird_arcwise_extra(cfg, recs, work):
    """Step 8 item 1: predictions on Arcwise's own wording for the 129 changed/duplicate items, scored vs Arcwise gold."""
    arc = {str(a["question_id"]): a for a in json.loads(Path(cfg["arcwise_json"]).read_text(encoding="utf-8"))}
    v1 = {str(e["question_id"]): e for e in json.loads((Path(cfg["bird_root"]) / "mini_dev_sqlite.json").read_text(encoding="utf-8"))}
    items = [{"pred": r.get("sql") or "", "gold": arc[r["id"][4:]]["SQL"], "db_id": arc[r["id"][4:]]["db_id"],
              "difficulty": v1[r["id"][4:]]["difficulty"], "question_id": r["id"][4:]} for r in recs]
    return run_bird_items(cfg, items, work, "arcextra")


def bird_per_item(cfg, work, tag, metric):
    import importlib, os, sqlite3 as _sq, time as _t
    ev = cfg["bird_eval_dir"]
    if ev not in sys.path:
        sys.path.insert(0, ev)
    eu = importlib.import_module("evaluation_utils")
    fn = importlib.import_module("evaluation_ex").calculate_ex if metric == "ex" else importlib.import_module("evaluation_f1").calculate_f1_score
    orig = _sq.connect

    def connect(*a, **k):
        c = orig(*a, **k); dl = _t.monotonic() + 30
        c.set_progress_handler(lambda: 1 if _t.monotonic() > dl else 0, 100_000)
        return c
    preds = json.loads((work / f"pred_{tag}.json").read_text(encoding="utf-8"))
    golds = (work / f"gold_{tag}.sql").read_text(encoding="utf-8").strip().splitlines()
    B = Path(cfg["bird_root"]) / "dev_databases"
    scores = []
    _sq.connect = connect
    try:
        for k in range(len(golds)):
            psql, _, db = preds[str(k)].split("	")
            gsql, gdb = golds[k].rsplit("	", 1)
            try:
                r = eu.execute_sql(psql, gsql, str(B / gdb / f"{gdb}.sqlite"), "SQLite", fn)
            except Exception:
                r = 0
            scores.append(float(r))
    finally:
        _sq.connect = orig
    with open(work / f"{metric}_{tag}.per_item.json", "w", encoding="utf-8") as f:
        json.dump(scores, f)
    return round(100 * sum(scores) / len(scores), 2) if scores else None


def power_points(run):
    pts = []
    for p in sorted((run / "power").glob("session*.log")):
        for line in open(p):
            if line.startswith("#"):
                continue
            ns, _, rest = line.partition(" ")
            m = VDD.search(rest)
            if m:
                pts.append((int(ns), int(m.group(1)) / 1000))
    pts.sort()
    return pts


def energy(pts, t0, t1):
    import bisect
    ts = [t for t, _ in pts]
    i, j = bisect.bisect_left(ts, t0), bisect.bisect_right(ts, t1)
    if i == 0 or j >= len(ts):
        return None
    seq = [(t0, pts[i - 1][1])] + pts[i:j] + [(t1, pts[j - 1][1])]
    return sum((b[0] - a[0]) / 1e9 * (a[1] + b[1]) / 2 for a, b in zip(seq, seq[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--idle-w", type=float)
    a = ap.parse_args()
    cfg = json.loads(a.cfg.read_text(encoding="utf-8"))
    gold = gold_nyc(cfg)
    conn = sqlite3.connect(f"{Path(cfg['taxi_db']).resolve().as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON"); conn.execute("PRAGMA mmap_size = 1073741824")
    arc_keep = set(json.loads(Path(cfg["arcwise_unchanged"]).read_text())["unchanged_question_ids"])

    if a.oracle:
        out = a.out; out.mkdir(parents=True, exist_ok=True)
        res = {}
        for name in ("nyc_eval", "nyc_pool"):
            recs = [{"rid": e["id"], "id": e["id"], "cond": name, "sql": e["sql"]} for e in rd(Path(cfg["experiment"]) / "data" / f"{name}.jsonl")]
            s = score_nyc(cfg, recs, gold, conn); res[name] = sum(x["strict"] for x in s) / len(s)
        cur = rd(Path(cfg["experiment"]) / "curated" / "curated.jsonl")
        s = score_nyc(cfg, [{"rid": e["id"], "id": e["id"], "cond": "cur", "sql": e["sql"]} for e in cur], gold, conn)
        res["curated"] = sum(x["strict"] for x in s) / len(s)
        sp = [{"rid": f"spider_dev_{i:04d}", "id": f"spider_dev_{i:04d}", "sql": e["query"]}
              for i, e in enumerate(json.loads((Path(cfg["spider_root"]) / "dev.json").read_text(encoding="utf-8")))]
        res["spider"] = run_spider(cfg, sp, out / "spider")
        bd = [{"rid": f"bird_minidev_{i:03d}", "id": f"bird_minidev_{i:03d}", "sql": e["SQL"]}
              for i, e in enumerate(json.loads((Path(cfg["bird_root"]) / "mini_dev_sqlite.json").read_text(encoding="utf-8")))]
        res["bird_v1"] = run_bird(cfg, bd, out / "bird")
        arc = {str(g["question_id"]): g["SQL"] for g in json.loads(Path(cfg["arcwise_json"]).read_text(encoding="utf-8"))}
        v1 = json.loads((Path(cfg["bird_root"]) / "mini_dev_sqlite.json").read_text(encoding="utf-8"))
        ba = [{"rid": f"bird_minidev_{i:03d}", "id": f"bird_minidev_{i:03d}", "sql": arc[str(e["question_id"])]}
              for i, e in enumerate(v1) if str(e["question_id"]) in arc]
        res["arcwise_gold_vs_arcwise"] = run_bird(cfg, ba, out / "bird", gold_src=cfg["arcwise_json"], keep=arc_keep, tag="arc")
        for k in ("bird_v1", "arcwise_gold_vs_arcwise"):
            res[k].pop("ids", None)
        (out / "oracle.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        bad = [k for k in ("nyc_eval", "nyc_pool", "curated") if res[k] != 1.0]
        bad += [f"spider.{m}" for m in ("ex", "ts", "em") if res["spider"].get(m) != 1.0]
        bad += [f"{b}.ex" for b in ("bird_v1", "arcwise_gold_vs_arcwise") if res[b].get("ex") is None or res[b]["ex"] < 99.99]
        bad += [f"{b}.f1" for b in ("bird_v1", "arcwise_gold_vs_arcwise") if res[b].get("f1") is None]
        if bad:
            print(f"ERROR oracle checks failed or returned null: {bad}"); sys.exit(2)
        print("ORACLE_ALL_PASS"); return

    run = a.run; out = run.parent.parent.parent / "build" / "scores" / run.name if a.out is None else a.out
    out.mkdir(parents=True, exist_ok=True)
    pts = power_points(run)
    summary = {}
    for f in sorted((run / "acc").glob("*/*.jsonl")):
        model, cond = f.parent.name, f.stem
        recs = rd(f)
        if not recs:
            continue
        key = f"{model}/{cond}"
        re_ex = sum(extract_sql(r["raw"])[0] != r["sql"] for r in recs)
        base = {"n": len(recs), "sql_extracted": sum(bool(r["sql"]) for r in recs), "cap_hits": sum(bool(r["truncated"]) for r in recs),
                "ctx_flags": sum(bool(r["ctx_flag"]) for r in recs), "extract_rules": dict(Counter(r["extract_rule"] for r in recs)),
                "reextract_mismatch": re_ex, "wall_s_mean": st.mean(r["wall_s"] for r in recs),
                "eval_tokens_mean": st.mean((r["eval_count"] or 0) for r in recs),
                "decode_tok_s_mean": st.mean(r["eval_count"] / (r["eval_duration"] / 1e9) for r in recs if r["eval_duration"]),
                "prefill_tok_s_mean": st.mean(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9) for r in recs if r.get("prompt_eval_duration")),
                "models": sorted({r["model"] for r in recs})}
        es = [energy(pts, r["t_send_ns"], r["t_recv_ns"]) for r in recs] if pts else []
        if es and all(e is not None for e in es):
            base["energy_j_mean"] = st.mean(es)
            if a.idle_w:
                base["energy_above_idle_j_mean"] = st.mean(e - a.idle_w * r["wall_s"] for e, r in zip(es, recs))
        if cond.startswith(("nyc", "cur", "lora", "determinism")):
            s = score_nyc(cfg, recs, gold, conn)
            base.update({"strict": sum(x["strict"] for x in s) / len(s), "lenient": sum(x["lenient"] for x in s) / len(s),
                         "classes": dict(Counter(x["class"] for x in s))})
            if cond == "nycsim" or cond.startswith("lora"):
                for role in ("seen", "unseen"):
                    sub = [x for x in s if x["role"] == role]
                    if sub:
                        base[f"strict_{role}"] = sum(x["strict"] for x in sub) / len(sub)
            (out / model).mkdir(exist_ok=True)
            with open(out / model / f"{cond}.details.jsonl", "w", encoding="utf-8") as fh:
                for x in s:
                    fh.write(json.dumps(x) + "\n")
        elif cond == "spider0":
            base["spider"] = run_spider(cfg, recs, out / model / "spider")
        elif cond == "birdarc":
            base["bird_arcwise_extra"] = bird_arcwise_extra(cfg, recs, out / model / cond)
        elif cond in ("bird0", "birdnat"):
            base["bird_v1"] = run_bird(cfg, recs, out / model / cond)
            base["bird_arcwise_unchanged"] = run_bird(cfg, recs, out / model / cond, gold_src=cfg["arcwise_json"], keep=arc_keep, tag="arc")
            base["bird_v1_on_arcwise_unchanged"] = run_bird(cfg, recs, out / model / cond, keep=arc_keep, tag="v1sub")
        summary[key] = base
        print(key, {k: v for k, v in base.items() if k in ("n", "strict", "sql_extracted", "cap_hits", "wall_s_mean")}, flush=True)
    for k in list(summary):
        if k.endswith("/bird0"):
            m = k.split("/")[0]; u = summary[k].get("bird_arcwise_unchanged", {}); x = summary.get(f"{m}/birdarc", {}).get("bird_arcwise_extra", {})
            if u.get("n") and x.get("n") and u.get("ex") is not None and x.get("ex") is not None:
                summary[k]["bird_arcwise_full"] = {"n": u["n"] + x["n"], "ex": round((u["ex"] * u["n"] + x["ex"] * x["n"]) / (u["n"] + x["n"]), 2),
                                                   "method": "369 unchanged items re-scored from v1 predictions + 129 items run on Arcwise wording"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    nulls = []
    for k, v in summary.items():
        for sub in ("spider", "bird_v1", "bird_arcwise_unchanged", "bird_v1_on_arcwise_unchanged", "bird_arcwise_extra"):
            if sub in v and v[sub].get("n"):
                nulls += [f"{k}:{sub}.{m}" for m in ("ex", "ts", "em", "f1") if m in v[sub] and v[sub][m] is None]
    if nulls:
        print(f"ERROR null metrics (evaluator failure): {nulls}"); sys.exit(2)
    print("SCORING_DONE no null metrics")


if __name__ == "__main__":
    main()
