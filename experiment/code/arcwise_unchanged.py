#!/usr/bin/env python3
"""List BIRD Mini-Dev v1 question_ids whose question and evidence are unchanged (whitespace-
normalised) in Arcwise-Plat-SQL (Jin et al.), i.e. where re-scoring v1 predictions against the
corrected gold SQL is valid. Found at prep: 127 of 498 items have changed question/evidence text.

    python arcwise_unchanged.py --v1 mini_dev_sqlite.json --arcwise arcwise_plat_sql_only_with_diff.json --out f.json
"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--v1"); ap.add_argument("--arcwise"); ap.add_argument("--out")
a = ap.parse_args()
n = lambda s: " ".join(str(s).split())
v1 = json.load(open(a.v1, encoding="utf-8")); arc = json.load(open(a.arcwise, encoding="utf-8"))
v = {}
for e in v1:
    v.setdefault(str(e["question_id"]), e)
dup = sorted({str(e["question_id"]) for e in v1 if sum(str(x["question_id"]) == str(e["question_id"]) for x in v1) > 1})
keep, changed = [], []
for x in arc:
    q = str(x["question_id"]); b = v[q]
    same = n(x["question"]) == n(b["question"]) and n(x["evidence"]) == n(b["evidence"]) and x["db_id"] == b["db_id"]
    (keep if same and q not in dup else changed).append(q)
sql_changed = sum(n(x["SQL"]) != n(v[str(x["question_id"])]["SQL"]) for x in arc if str(x["question_id"]) in keep)
json.dump({"unchanged_question_ids": keep, "changed_or_duplicate_question_ids": changed, "v1_duplicate_question_ids": dup,
           "n_unchanged": len(keep), "n_changed_or_dup": len(changed), "gold_sql_changed_within_unchanged": sql_changed},
          open(a.out, "w"), indent=1)
print(len(keep), len(changed), dup, sql_changed)
