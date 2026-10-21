#!/usr/bin/env python3
"""Automatic checks for the curated (AI-drafted) NYC set (PLAN 3.3 steps 3, 4, 6, 9).

For every drafted item: execute gold A and gold B on the taxi DB with eval_exec.py's executor
and normalisation; accept only if both agree (eval_exec same_rows, ordered flag from gold A),
the query runs in < 10 s, the result is non-empty and non-degenerate, and any ORDER BY ... LIMIT
cut has no tie (rank_sql check, as gen_questions.has_tie). Each accepted item gets a structure
tag: "template-like" if its masked SQL skeleton equals any of the 26 templates', else "novel".
The 15 exploratory seed items (handwritten.jsonl) go through the same checks, flagged "seed".

Writes (not frozen; no hash goes to data.lock.json here):
    <out>/curated.jsonl          accepted items, label "curated (AI-drafted)"
    <out>/curated_check.json     every item: status, reasons, timings, A/B results preview
    <spotcheck>                  20 accepted items, seeded random, for the team's spot-check

    python curate_check.py --db taxi.db --drafts experiment/curated/drafts --seeds handwritten.jsonl \
        --out experiment/curated --spotcheck experiment/curated_spotcheck.md
"""
import argparse
import json
import random
import re
import sqlite3
import time
from pathlib import Path

from eval_exec import execute, norm, same_rows
from gen_questions import is_degenerate
from templates import SLOT_TYPES, TEMPLATES

LABEL = "curated (AI-drafted)"
KEYWORDS = set("""select from where join on and or not in is null group by order having limit as distinct case
when then else end count sum avg min max cast integer real text strftime date julianday between like exists
union intersect except all asc desc inner left outer cross round abs iif coalesce ifnull nullif substr""".split())


def skeleton(sql):
    s = re.sub(r"'(?:[^']|'')*'", " ? ", sql)
    s = re.sub(r"\b\d+(\.\d+)?\b", " ? ", s)
    toks = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\?|[^\sA-Za-z_0-9]", s)
    out = [t.lower() if t.lower() in KEYWORDS else ("?" if t == "?" else ("ID" if re.match(r"[A-Za-z_]", t) else t))
           for t in toks]
    return " ".join(out)


def template_skeletons():
    class Ctx(dict):
        pass
    ctx = {"boroughs": ["Manhattan"], "zones": ["Midtown Center"], "dates": ["2024-01-02"]}
    rng = random.Random(0)
    sk = {}
    for t in TEMPLATES:
        vals = {name: SLOT_TYPES[kind](rng, ctx)[1] for name, kind in t.slots.items()}
        sk[skeleton(t.sql.format(**vals))] = t.id
    return sk


def run(conn, sql, timeout=10.0):
    t0 = time.perf_counter()
    rows, err = execute(conn, sql, timeout)
    return rows, err, time.perf_counter() - t0


def preview(rows, n=5):
    if rows is None:
        return None
    return {"n_rows": len(rows), "head": rows[:n]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--drafts", type=Path, required=True)
    ap.add_argument("--seeds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spotcheck", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261016)
    args = ap.parse_args()

    rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    qs = rd(args.drafts / "questions.jsonl")
    ga = {g["id"]: g for g in rd(args.drafts / "goldA.jsonl")}
    gb = {g["id"]: g for g in rd(args.drafts / "goldB.jsonl")}
    for s in rd(args.seeds):
        qs.append({"id": s["id"], "category": "seed", "question": s["question"], "seed": True})
        ga[s["id"]] = {"id": s["id"], "sql": s["sql"], "ordered": bool(s.get("ordered", False))}

    conn = sqlite3.connect(f"{args.db.resolve().as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA mmap_size = 1073741824")
    tsk = template_skeletons()
    checks, accepted = [], []
    for q in qs:
        a, b = ga[q["id"]], gb.get(q["id"])
        reasons = []
        ordered = bool(a.get("ordered", False))
        ra, ea, ta = run(conn, a["sql"])
        rb, eb, tb = run(conn, b["sql"]) if b else (None, "no gold B", 0.0)
        if ea:
            reasons.append(f"gold A error: {ea}")
        if eb:
            reasons.append(f"gold B error: {eb}")
        if ta >= 10 or tb >= 10:
            reasons.append("slow (>= 10 s)")
        # a single 0 answer counts as degenerate for curated items (weak test), whatever allow_zero says
        if ra is not None and is_degenerate(ra, False):
            reasons.append("empty or degenerate result")
        if ra is not None and rb is not None and not same_rows(rb, ra, ordered):
            reasons.append("gold A and gold B disagree")
        if a.get("rank_sql"):
            k = int(a.get("k", 1))
            ms = [r[0] for r in conn.execute(a["rank_sql"].format(n=k + 1)).fetchall()]
            if len(set(ms)) != len(ms):
                reasons.append(f"tie at the LIMIT {k} cut")
        tag_t = tsk.get(skeleton(a["sql"]))
        rec = {"id": q["id"], "category": q["category"], "seed": bool(q.get("seed")), "question": q["question"],
               "sql_a": a["sql"], "sql_b": b["sql"] if b else None, "ordered": ordered,
               "status": "accepted" if not reasons else "rejected", "reasons": reasons,
               "time_a_s": round(ta, 3), "time_b_s": round(tb, 3),
               "structure": "template-like" if tag_t else "novel", "matched_template": tag_t,
               "result_a": preview(ra), "result_b": preview(rb)}
        checks.append(rec)
        if not reasons:
            raw = [list(r) for r in conn.execute(a["sql"]).fetchall()]
            accepted.append({"id": q["id"], "split": "curated", "label": LABEL, "status": "draft, not frozen",
                             "category": q["category"], "seed": bool(q.get("seed")), "structure": rec["structure"],
                             "template_id": None, "question": q["question"], "sql": a["sql"], "sql_b": b["sql"],
                             "ordered": ordered, "answer": raw})
    conn.close()

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "curated.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in accepted:
            f.write(json.dumps(r) + "\n")
    summ = {"drafted": len([c for c in checks if not c["seed"]]), "seed_items": len([c for c in checks if c["seed"]]),
            "accepted": len(accepted), "accepted_new": len([a for a in accepted if not a["seed"]]),
            "accepted_seed": len([a for a in accepted if a["seed"]]),
            "template_like": len([a for a in accepted if a["structure"] == "template-like"]),
            "rejected": [(c["id"], c["reasons"]) for c in checks if c["status"] == "rejected"]}
    (args.out / "curated_check.json").write_text(json.dumps({"summary": summ, "items": checks}, indent=1),
                                                 encoding="utf-8", newline="\n")
    print(json.dumps(summ, indent=1))

    pick = sorted(random.Random(args.seed).sample(range(len(accepted)), min(10, len(accepted))))
    by = {c["id"]: c for c in checks}
    L = [f"# Curated (AI-drafted) set: team spot-check (D12)", "",
         f"Status: **draft, not frozen**. {len(accepted)} items passed the automatic checks "
         f"({summ['accepted_new']} new, {summ['accepted_seed']} exploratory seed items). "
         f"These 10 were drawn with `random.Random({args.seed}).sample` over the accepted list "
         f"(script `experiment/code/curate_check.py`).", "",
         "For each item please check: is the question unambiguous, does the gold SQL answer it as worded, "
         "and is the answer plausible? Mark OK / FIX (say what) / DROP. "
         "Gold A was written by the drafting agent; gold B was written independently in a separate session "
         "that saw only the schema and the question (see `experiment/curated/DRAFTING.md`). Both return the same result.", ""]
    for n, i in enumerate(pick, 1):
        a = accepted[i]
        c = by[a["id"]]
        res = c["result_a"]
        head = "; ".join(str(r) for r in res["head"])
        L += [f"## {n}. `{a['id']}` ({a['category']}{', seed' if a['seed'] else ''}, {a['structure']}"
              f"{', ordered' if a['ordered'] else ''})", "",
              f"**Question:** {a['question']}", "", "**Gold SQL (A):**", "", "```sql", a["sql"], "```", "",
              "**Independent gold SQL (B):**", "", "```sql", a["sql_b"], "```", "",
              f"**Result preview:** {res['n_rows']} row(s): {head}", "", "Verdict: ______", ""]
    args.spotcheck.write_text("\n".join(L), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
