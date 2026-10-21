#!/usr/bin/env python3
"""Build the NYC eval set, the example pool and the four held-out template folds (PLAN 3, 4.2).

All 1,193 template instances (train + test_seen + test_unseen from gen_questions.py) are
pooled. Per template (in templates.py order, instances sorted by id), one random.Random(seed)
stream shuffles the instances; the first m_t = min(20, n_t // 2) become eval items and the
rest go to the pool. Each template is assigned to exactly one fold (FOLDS below, PLAN 4.2).

Writes:
    <out-data>/nyc_eval.jsonl   eval items, with "fold" (the fold in which the item is unseen)
    <out-data>/nyc_pool.jsonl   pool items, with "fold" (the fold whose pool EXCLUDES them)
    <out-config>/folds.json     fold -> templates, counts, ids; pool ids per fold; seen/unseen pairing

Usage:
    python make_folds.py --questions data/questions --out-data experiment/data --out-config experiment/config
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from templates import TEMPLATES

FOLDS = {
    "F1": ["highest_avg_fare_zone_min_pickups", "cross_borough_trips_hour", "avg_speed_borough",
           "busiest_date_borough", "zones_without_pickups", "above_avg_fare_zone"],
    "F2": ["count_trips_borough", "avg_tip_by_borough", "busiest_pickup_zone_hour", "pct_payment_borough",
           "count_borough_to_borough", "max_total_date", "distinct_dropoffs_from_zone"],
    "F3": ["revenue_borough_dow", "longest_trip_per_payment_borough", "peak_hour_zone",
           "avg_tolls_borough_to_borough", "count_passengers_miles", "avg_tip_pct_hour_credit",
           "airport_dropoffs_from_borough"],
    "F4": ["avg_fare_zone", "avg_duration_borough", "top_k_dropoff_zones_from_borough", "top_tip_borough",
           "trips_per_payment_borough", "avg_passengers_miles"],
}
FOLD_ORDER = ["F1", "F2", "F3", "F4"]
# PLAN 4.2 expected counts (eval items per fold, pool size per fold)
EXPECTED = {"F1": (112, 532), "F2": (121, 546), "F3": (111, 576), "F4": (105, 578)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--out-data", type=Path, required=True)
    ap.add_argument("--out-config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=20)
    args = ap.parse_args()

    all_t = [t.id for t in TEMPLATES]
    assigned = [t for f in FOLD_ORDER for t in FOLDS[f]]
    assert sorted(assigned) == sorted(all_t), "every template must be in exactly one fold"
    fold_of = {t: f for f in FOLD_ORDER for t in FOLDS[f]}

    items = []
    for name in ("train", "test_seen", "test_unseen"):
        with open(args.questions / f"{name}.jsonl", encoding="utf-8") as fh:
            items += [json.loads(l) for l in fh if l.strip()]
    assert len(items) == 1193, len(items)

    by_t = {t: sorted([e for e in items if e["template_id"] == t], key=lambda e: e["id"]) for t in all_t}
    rng = random.Random(args.seed)
    eval_items, pool_items = [], []
    for t in all_t:
        insts = list(by_t[t])
        rng.shuffle(insts)
        m = min(args.max_eval, len(insts) // 2)
        for i, ex in enumerate(insts):
            rec = {"id": ex["id"], "template_id": t, "fold": fold_of[t],
                   "set": "eval" if i < m else "pool", "orig_split": ex["split"],
                   "question": ex["question"], "sql": ex["sql"], "ordered": ex["ordered"],
                   "answer": ex["answer"]}
            (eval_items if i < m else pool_items).append(rec)

    args.out_data.mkdir(parents=True, exist_ok=True)
    args.out_config.mkdir(parents=True, exist_ok=True)
    for name, recs in (("nyc_eval", eval_items), ("nyc_pool", pool_items)):
        with open(args.out_data / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")

    folds = {}
    for k, f in enumerate(FOLD_ORDER):
        prev = FOLD_ORDER[k - 1]  # cyclic: F1 uses F4 as its seen items
        ev = [e for e in eval_items if e["fold"] == f]
        pool = [p for p in pool_items if p["fold"] != f]
        folds[f] = {
            "heldout_templates": FOLDS[f],
            "eval_counts": dict(Counter(e["template_id"] for e in ev)),
            "n_eval": len(ev),
            "n_pool": len(pool),
            "unseen_ids": [e["id"] for e in ev],
            "seen_from_fold": prev,
            "seen_ids": [e["id"] for e in eval_items if e["fold"] == prev],
            "pool_ids": [p["id"] for p in pool],
        }
        exp = EXPECTED[f]
        assert (len(ev), len(pool)) == exp, f"{f}: got {(len(ev), len(pool))}, PLAN says {exp}"
    meta = {"seed": args.seed, "max_eval": args.max_eval, "n_eval": len(eval_items),
            "n_pool": len(pool_items), "folds": folds}
    (args.out_config / "folds.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    print(f"eval {len(eval_items)}, pool {len(pool_items)}")
    for f in FOLD_ORDER:
        print(f"{f}: eval {folds[f]['n_eval']}, pool {folds[f]['n_pool']}, seen from {folds[f]['seen_from_fold']}")
    assert len(eval_items) == 449 and len(pool_items) == 744


if __name__ == "__main__":
    main()
