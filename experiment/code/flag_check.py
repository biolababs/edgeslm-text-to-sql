#!/usr/bin/env python3
"""Throttling check over a time window of the 1 s sysfs sampler log (gpu_sampler.sh), per the step 8
sign-off (item 3): throttling = GPU clock below 1,020 MHz or tj >= 95 C. INA3221 crit_alarm samples are
counted and reported but do not flag. Exit 1 if throttled, 2 (with an ERROR line) if the log is missing. Stdlib only.

    python3 flag_check.py --gpu gpu_s3.tsv --t0 <ns> --t1 <ns>
"""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True); ap.add_argument("--t0", type=int, required=True); ap.add_argument("--t1", type=int, required=True)
ap.add_argument("--min-hz", type=int, default=1_020_000_000); ap.add_argument("--max-tj", type=float, default=95.0)
a = ap.parse_args()
import os
if not a.gpu or not os.path.exists(a.gpu):
    print(json.dumps({"error": f"GPU sampler log not found: {a.gpu!r}"}))
    print(f"ERROR flag_check: GPU sampler log not found: {a.gpu!r}", file=sys.stderr)
    sys.exit(2)  # distinct from 1 (throttled): a missing file is never read as throttling
rows = []
for line in open(a.gpu):
    f = line.rstrip("\n").split("\t")
    if f[0].isdigit() and a.t0 <= int(f[0]) <= a.t1:
        rows.append(f)
if not rows:
    print(json.dumps({"samples": 0, "error": "no sampler rows in window"})); print("ERROR flag_check: no GPU sampler rows in the window", file=sys.stderr); sys.exit(2)
min_hz = min(int(f[1]) for f in rows)
max_tj = max(int(f[4]) for f in rows) / 1000
crit = sum(any(x == "1" for x in f[5].split(",")) for f in rows)
thr = min_hz < a.min_hz or max_tj >= a.max_tj
print(json.dumps({"samples": len(rows), "min_gpu_hz": min_hz, "max_tj_c": max_tj, "crit_alarm_samples": crit, "throttled": thr}))
sys.exit(1 if thr else 0)
