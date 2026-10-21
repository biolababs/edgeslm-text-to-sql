#!/usr/bin/env python3
"""Pilot LoRA smoke requests (PLAN 10): 10 pool items the adapter was trained on, and 10 fold-f
eval items (unseen templates), both zero-shot with the common prompt (prompts.messages_common,
the same function build_requests.py uses on the PC). Stdlib only.

    python3 make_lora_requests.py --train-log train/f1/train_log.json --pool nyc_pool.jsonl \
        --nyc0 requests/nyc0.jsonl --schema schema.sql --fold F1 --out requests/pilot_lora.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from prompts import messages_common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--nyc0", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    seen = json.loads(a.train_log.read_text())["seen_ids"]
    pool = {p["id"]: p for p in map(json.loads, open(a.pool, encoding="utf-8"))}
    schema = a.schema.read_text(encoding="utf-8")
    rng = random.Random(11)
    tr = rng.sample(sorted(set(seen)), 10)
    ev = [r for r in map(json.loads, open(a.nyc0, encoding="utf-8")) if r["fold"] == a.fold]
    ev = rng.sample(ev, 10)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for i in tr:
            p = pool[i]
            f.write(json.dumps({"rid": f"{i}|train", "id": i, "cond": "lora_trainpool0", "set": "pool",
                                "template_id": p["template_id"], "fold": p["fold"],
                                "messages": messages_common(schema, p["question"])}) + "\n")
        for r in ev:
            r = dict(r, rid=f"{r['id']}|evalF", cond="lora_eval0")
            f.write(json.dumps(r) + "\n")
    print(f"wrote {a.out}: 10 trained pool items + 10 {a.fold} eval items")


if __name__ == "__main__":
    main()
