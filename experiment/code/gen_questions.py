#!/usr/bin/env python3
"""Generate question/SQL pairs with gold answers from the taxi database.

Splits written to --out-dir (JSONL, one example per line):
    train.jsonl            instances of the non-held-out templates
    test_seen.jsonl        unseen instances of the training templates
    test_unseen.jsonl      every instance of the held-out templates
    test_handwritten.jsonl the hand-written questions, with gold answers filled in

Each example: id, split, template_id, question, sql, ordered, answer (rows).

Usage:
    python gen_questions.py --db data/taxi.db --per-template 60
"""
import argparse
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

from templates import DEFAULT_HELDOUT, SLOT_TYPES, TEMPLATES, build_context


def run(conn, sql):
    return [list(r) for r in conn.execute(sql).fetchall()]


def is_degenerate(rows, allow_zero):
    if not rows:
        return True
    if len(rows) == 1 and len(rows[0]) == 1:
        v = rows[0][0]
        return v is None or (v == 0 and not allow_zero)
    return False


def has_tie(conn, tmpl, sql_vals):
    """True if the top-k cut of an ORDER BY ... LIMIT query is ambiguous."""
    if not tmpl.rank_sql:
        return False
    k = int(sql_vals.get("k", 1))
    metrics = [r[0] for r in run(conn, tmpl.rank_sql.format(n=k + 1, **sql_vals))]
    return len(set(metrics)) != len(metrics)


def instantiate(conn, tmpl, ctx, rng, n_wanted, max_tries):
    out, seen = [], set()
    for _ in range(max_tries):
        if len(out) >= n_wanted:
            break
        nl_vals, sql_vals = {}, {}
        for name, kind in tmpl.slots.items():
            nl_vals[name], sql_vals[name] = SLOT_TYPES[kind](rng, ctx)
        sql = tmpl.sql.format(**sql_vals)
        if sql in seen:
            if not tmpl.slots:
                break
            continue
        seen.add(sql)
        rows = run(conn, sql)
        if is_degenerate(rows, tmpl.allow_zero) or has_tie(conn, tmpl, sql_vals):
            continue
        out.append({
            "template_id": tmpl.id,
            "question": rng.choice(tmpl.questions).format(**nl_vals),
            "sql": sql,
            "ordered": tmpl.ordered,
            "answer": rows,
        })
    return out


def load_handwritten(conn, path):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            rows = run(conn, ex["sql"])
            if not rows:
                print(f"  warning: hand-written {ex['id']} returns no rows")
            examples.append({
                "id": ex["id"],
                "split": "test_handwritten",
                "template_id": None,
                "question": ex["question"],
                "sql": ex["sql"],
                "ordered": bool(ex.get("ordered", False)),
                "answer": rows,
            })
    return examples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("data/taxi.db"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/questions"))
    ap.add_argument("--per-template", type=int, default=60, help="target instances per template")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="fraction of training-template instances moved to test_seen")
    ap.add_argument("--holdout", nargs="*", default=DEFAULT_HELDOUT,
                    help="template ids reserved for test_unseen")
    ap.add_argument("--handwritten", type=Path, default=Path(__file__).with_name("handwritten.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    known = {t.id for t in TEMPLATES}
    unknown = set(args.holdout) - known
    if unknown:
        ap.error(f"unknown template ids in --holdout: {sorted(unknown)}")

    conn = sqlite3.connect(f"{args.db.resolve().as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA mmap_size = 1073741824")  # ~5x faster on the scan-heavy templates
    ctx = build_context(conn)
    rng = random.Random(args.seed)

    splits = {"train": [], "test_seen": [], "test_unseen": []}
    for tmpl in TEMPLATES:
        insts = instantiate(conn, tmpl, ctx, rng, args.per_template, args.per_template * 20)
        if len(insts) < args.per_template:
            print(f"  note: {tmpl.id} produced {len(insts)}/{args.per_template} instances")
        if tmpl.id in args.holdout:
            for ex in insts:
                ex["split"] = "test_unseen"
            splits["test_unseen"] += insts
            continue
        rng.shuffle(insts)
        n_test = round(len(insts) * args.test_frac)
        for i, ex in enumerate(insts):
            ex["split"] = "test_seen" if i < n_test else "train"
            splits[ex["split"]].append(ex)

    if args.handwritten.exists():
        splits["test_handwritten"] = load_handwritten(conn, args.handwritten)
    conn.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, exs in splits.items():
        for i, ex in enumerate(exs):
            ex.setdefault("id", f"{name}_{i:05d}")
        with open(args.out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in exs:
                f.write(json.dumps({"id": ex["id"], **{k: v for k, v in ex.items() if k != "id"}}) + "\n")

    print(f"\n{'split':<18}{'examples':>9}{'templates':>11}")
    for name, exs in splits.items():
        n_t = len({ex["template_id"] for ex in exs if ex["template_id"]})
        print(f"{name:<18}{len(exs):>9}{n_t:>11}")
    print(f"\nheld-out templates: {', '.join(args.holdout)}")
    per_t = Counter(ex["template_id"] for exs in splits.values() for ex in exs if ex["template_id"])
    print("instances per template:", dict(sorted(per_t.items())))


if __name__ == "__main__":
    main()
