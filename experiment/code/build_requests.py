#!/usr/bin/env python3
"""Pre-render every request (chat messages) for every condition, on the PC (PLAN 4, 5).

The Jetson runner only sends these files; it never builds prompts, so the exact text each
model saw is fixed here and hashed. One JSONL per condition, one record per request:
    {"rid", "id", "cond", "set", "messages", ...metadata}

Conditions (PLAN 4.1):
    nyc0      449 eval items, zero-shot
    nycsim    898 = per fold g: fold-g items (unseen) + fold-(g-1) items (seen), S-sim from pool g
    nycfix    449, each item in its own held-out fold g, S-fixed from pool g
    cur0 / cursim / curfix   curated items, 0-shot / S-sim over the full pool / fold-1 fixed set
    spider0   1,034 Spider dev, zero-shot
    bird0     500 BIRD Mini-Dev v1, zero-shot with hint
    nycnat / curnat / birdnat   XiYan native prompt (M5 only)
    warmup    5 pool items, zero-shot (untimed warm-up)
With --pilot, also writes pilot_*.jsonl subsets as defined in PLAN 10.

    python build_requests.py --data experiment/data --config experiment/config --schema schema.sql \
        --public data/public --curated experiment/curated/curated.jsonl --out experiment/requests [--pilot]
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from prompts import TfIdf, fixed_shots, messages_common, messages_xiyan_native

FOLD_ORDER = ["F1", "F2", "F3", "F4"]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write(out, name, recs):
    p = out / f"{name}.jsonl"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--public", type=Path, required=True)
    ap.add_argument("--curated", type=Path)
    ap.add_argument("--xiyan-template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--arcwise", type=Path, help="Arcwise-Plat-SQL json (corrected Mini-Dev)")
    ap.add_argument("--arcwise-unchanged", type=Path, help="config/arcwise_unchanged.json")
    args = ap.parse_args()

    schema = args.schema.read_text(encoding="utf-8")
    xiyan = args.xiyan_template.read_text(encoding="utf-8")
    ev = read_jsonl(args.data / "nyc_eval.jsonl")
    pool = read_jsonl(args.data / "nyc_pool.jsonl")
    folds = json.loads((args.config / "folds.json").read_text(encoding="utf-8"))["folds"]
    by_id = {e["id"]: e for e in ev}
    pools = {f: [p for p in pool if p["fold"] != f] for f in FOLD_ORDER}
    cur = read_jsonl(args.curated) if args.curated and args.curated.exists() else []
    args.out.mkdir(parents=True, exist_ok=True)
    conds = {}

    def base(e, cond, s, rid=None):
        return {"rid": rid or e["id"], "id": e["id"], "cond": cond, "set": s,
                "template_id": e.get("template_id"), "fold": e.get("fold")}

    conds["nyc0"] = [{**base(e, "nyc0", "nyc"), "messages": messages_common(schema, e["question"])} for e in ev]

    sim, fix = [], []
    for k, g in enumerate(FOLD_ORDER):
        ret = TfIdf(pools[g])
        for role, ids in (("unseen", folds[g]["unseen_ids"]), ("seen", folds[g]["seen_ids"])):
            for i in ids:
                e = by_id[i]
                shots = ret.top_k(e["question"], 3)
                rec = base(e, "nycsim", "nyc", f"{i}|{g}|{role}")
                rec.update({"pool_fold": g, "role": role, "shot_ids": [s["id"] for s, _ in shots],
                            "shot_sims": [round(x, 6) for _, x in shots],
                            "top1_same_template": shots[-1][0]["template_id"] == e["template_id"],
                            "messages": messages_common(schema, e["question"],
                                                        shots=[(s["question"], s["sql"]) for s, _ in shots])})
                sim.append(rec)
        fs = fixed_shots(pools[g], seed=0)
        for i in folds[g]["unseen_ids"]:
            e = by_id[i]
            rec = base(e, "nycfix", "nyc", f"{i}|{g}|fixed")
            rec.update({"pool_fold": g, "role": "unseen", "shot_ids": [s["id"] for s in fs],
                        "messages": messages_common(schema, e["question"], shots=[(s["question"], s["sql"]) for s in fs])})
            fix.append(rec)
    conds["nycsim"], conds["nycfix"] = sim, fix

    if cur:
        full = TfIdf(pool)
        f1 = fixed_shots(pools["F1"], seed=0)
        conds["cur0"] = [{**base(e, "cur0", "curated"), "messages": messages_common(schema, e["question"])} for e in cur]
        conds["cursim"] = []
        for e in cur:
            shots = full.top_k(e["question"], 3)
            conds["cursim"].append({**base(e, "cursim", "curated"), "pool_fold": "all",
                                    "shot_ids": [s["id"] for s, _ in shots],
                                    "messages": messages_common(schema, e["question"],
                                                                shots=[(s["question"], s["sql"]) for s, _ in shots])})
        conds["curfix"] = [{**base(e, "curfix", "curated"), "pool_fold": "F1", "shot_ids": [s["id"] for s in f1],
                            "messages": messages_common(schema, e["question"], shots=[(s["question"], s["sql"]) for s in f1])}
                           for e in cur]
        conds["curnat"] = [{**base(e, "curnat", "curated"), "messages": messages_xiyan_native(xiyan, schema, e["question"])}
                           for e in cur]

    spider = read_jsonl(args.public / "spider_dev.jsonl")
    bird = read_jsonl(args.public / "bird_minidev.jsonl")
    conds["spider0"] = [{"rid": e["id"], "id": e["id"], "cond": "spider0", "set": "spider", "db_id": e["db_id"],
                         "messages": messages_common(e["schema"], e["question"])} for e in spider]
    conds["bird0"] = [{"rid": e["id"], "id": e["id"], "cond": "bird0", "set": "bird", "db_id": e["db_id"],
                       "difficulty": e["difficulty"], "question_id": e["question_id"],
                       "messages": messages_common(e["schema"], e["question"], e["evidence"])} for e in bird]
    conds["nycnat"] = [{**base(e, "nycnat", "nyc"), "messages": messages_xiyan_native(xiyan, schema, e["question"])} for e in ev]
    conds["birdnat"] = [{"rid": e["id"], "id": e["id"], "cond": "birdnat", "set": "bird", "db_id": e["db_id"],
                         "difficulty": e["difficulty"], "question_id": e["question_id"],
                         "messages": messages_xiyan_native(xiyan, e["schema"], e["question"], e["evidence"])} for e in bird]
    if args.arcwise:
        # Step 8 sign-off item 1: inference on the corrected Mini-Dev items whose question/evidence
        # differ from v1 (plus the 2 duplicated v1 ids), with Arcwise's own question and evidence.
        unch = set(json.loads(args.arcwise_unchanged.read_text())["unchanged_question_ids"])
        schema_of = {e["db_id"]: e["schema"] for e in bird}
        diff_of = {str(e["question_id"]): e["difficulty"] for e in bird}
        conds["birdarc"] = [{"rid": f"arc_{a['question_id']}", "id": f"arc_{a['question_id']}", "cond": "birdarc",
                             "set": "bird_arcwise", "db_id": a["db_id"], "question_id": str(a["question_id"]),
                             "difficulty": diff_of.get(str(a["question_id"])),
                             "messages": messages_common(schema_of[a["db_id"]], a["question"], a["evidence"])}
                            for a in json.loads(args.arcwise.read_text(encoding="utf-8")) if str(a["question_id"]) not in unch]
    # LoRA evaluation subsets (PLAN 7): fold f items (unseen) + fold f-1 items (seen)
    for k, g in enumerate(FOLD_ORDER):
        prev = FOLD_ORDER[k - 1]
        conds[f"lora{k + 1}0"] = [dict(r, rid=f"{r['id']}|{g}|{'unseen' if r['fold'] == g else 'seen'}", cond=f"lora{k + 1}0",
                                       pool_fold=g, role="unseen" if r["fold"] == g else "seen")
                                  for r in conds["nyc0"] if r["fold"] in (g, prev)]
        conds[f"lora{k + 1}sim"] = [dict(r, cond=f"lora{k + 1}sim") for r in sim if r["pool_fold"] == g]
    wu = sorted(pool, key=lambda p: p["id"])
    wu = random.Random(5).sample(wu, 5)
    conds["warmup"] = [{"rid": f"warmup_{p['id']}", "id": p["id"], "cond": "warmup", "set": "pool",
                        "messages": messages_common(schema, p["question"])} for p in wu]

    manifest = {}
    for name, recs in conds.items():
        p = write(args.out, name, recs)
        manifest[name] = {"n": len(recs), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    if args.pilot:
        rng = random.Random(10)  # pilot selection only; never evidence
        pil = {}
        pil["nyc0"] = [r for g in FOLD_ORDER for r in rng.sample([x for x in conds["nyc0"] if x["fold"] == g], 2)]
        pil["nycsim"] = (rng.sample([x for x in sim if x["role"] == "seen"], 4)
                         + rng.sample([x for x in sim if x["role"] == "unseen"], 4))
        pil["nycfix"] = rng.sample(fix, 4)
        if cur:
            cs = sorted(rng.sample(range(len(cur)), 5))
            pil["cur0"] = [conds["cur0"][i] for i in cs]
            pil["curnat"] = [conds["curnat"][i] for i in cs]
        pil["spider0"] = rng.sample(conds["spider0"], 10)
        pil["bird0"] = rng.sample(conds["bird0"], 10)
        pil["nycnat"] = [conds["nycnat"][[x["rid"] for x in conds["nyc0"]].index(r["rid"])] for r in pil["nyc0"][:5]]
        bn = {x["rid"]: x for x in conds["birdnat"]}
        pil["birdnat"] = [bn[r["rid"]] for r in pil["bird0"][:5]]
        # determinism: 20 NYC items (0-shot) run twice back to back
        pil["determinism"] = rng.sample(conds["nyc0"], 20)
        for name, recs in pil.items():
            p = write(args.out, f"pilot_{name}", recs)
            manifest[f"pilot_{name}"] = {"n": len(recs), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    sim_seen = [r for r in sim if r["role"] == "seen"]
    print(json.dumps({k: v["n"] for k, v in manifest.items()}))
    print("S-sim seen items whose top-1 shot shares their template: "
          f"{sum(r['top1_same_template'] for r in sim_seen)}/{len(sim_seen)}; unseen: "
          f"{sum(r['top1_same_template'] for r in sim if r['role'] == 'unseen')}")


if __name__ == "__main__":
    main()
