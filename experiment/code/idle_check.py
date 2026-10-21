#!/usr/bin/env python3
"""Idle verification and baseline summary from a power_log.py file (PLAN 9.2). Stdlib only.

Criteria (all must hold): GR3D_FREQ 0% in >= 99% of samples; mean CPU < 5%; VDD_IN sd < 0.2 W;
no process except the samplers above 1% CPU on average (top, first iteration skipped);
`ollama ps` empty; listening sockets limited to ssh(22), rpcbind(111), resolved(53),
cups(631) and ollama on 127.0.0.1:11434; swap used < 50 MB.

    python3 idle_check.py --power p.log --top top.txt --ollama-ps ps.txt --ss ss.txt --free free.txt
    python3 idle_check.py --power p.log --summary-only
"""
import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict

VDD = re.compile(r"VDD_IN (\d+)mW")
GR3D = re.compile(r"GR3D_FREQ (\d+)%")
CPU = re.compile(r"CPU \[([^\]]*)\]")
TJ = re.compile(r"tj@([\d.]+)C")
RAM = re.compile(r"RAM (\d+)/")
SWAP = re.compile(r"SWAP (\d+)/")
SAMPLERS = ("tegrastats", "power_log", "top", "python3", "sshd", "bash", "sleep", "gpu_sampler")
ALLOWED_PORTS = {"22", "111", "53", "631", "54"}


def parse_power(path):
    rows = []
    for line in open(path):
        if line.startswith("#"):
            continue
        ns, _, rest = line.partition(" ")
        v = VDD.search(rest)
        if not v:
            continue
        cpus = [int(x.split("%")[0]) for x in CPU.search(rest).group(1).split(",") if "%" in x] if CPU.search(rest) else []
        rows.append({"ns": int(ns), "vdd_w": int(v.group(1)) / 1000, "gr3d": int(GR3D.search(rest).group(1)),
                     "cpu": sum(cpus) / len(cpus) if cpus else 0.0, "tj": float(TJ.search(rest).group(1)),
                     "ram": int(RAM.search(rest).group(1)), "swap": int(SWAP.search(rest).group(1))})
    return rows


def summary(rows):
    ts = [r["ns"] for r in rows]
    v = [r["vdd_w"] for r in rows]
    return {"n": len(rows), "duration_s": (ts[-1] - ts[0]) / 1e9, "rate_hz": (len(ts) - 1) / ((ts[-1] - ts[0]) / 1e9),
            "ts_monotonic": all(b > a for a, b in zip(ts, ts[1:])),
            "vdd_in_median_w": st.median(v), "vdd_in_mean_w": st.mean(v), "vdd_in_sd_w": st.pstdev(v),
            "gr3d_zero_share": sum(r["gr3d"] == 0 for r in rows) / len(rows),
            "cpu_mean_pct": st.mean(r["cpu"] for r in rows), "tj_median_c": st.median(r["tj"] for r in rows),
            "tj_max_c": max(r["tj"] for r in rows), "ram_mb_median": st.median(r["ram"] for r in rows),
            "swap_mb_max": max(r["swap"] for r in rows)}


def top_procs(path):
    iters, cur = [], None
    for line in open(path):
        if line.startswith("top -"):
            cur = defaultdict(float); iters.append(cur); continue
        f = line.split()
        if cur is not None and len(f) >= 12 and f[0].isdigit():
            try:
                cur[f[11]] += float(f[8])
            except ValueError:
                pass
    iters = iters[1:] if len(iters) > 1 else iters
    tot = defaultdict(float)
    for it in iters:
        for k, x in it.items():
            tot[k] += x
    return {k: x / len(iters) for k, x in tot.items()}, len(iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", required=True)
    ap.add_argument("--top"); ap.add_argument("--ollama-ps"); ap.add_argument("--ss"); ap.add_argument("--free")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    s = summary(parse_power(a.power))
    if a.summary_only:
        print(json.dumps(s, indent=1)); return 0
    checks = {"gr3d_zero_ge_99pct": s["gr3d_zero_share"] >= 0.99, "cpu_mean_lt_5pct": s["cpu_mean_pct"] < 5,
              "vdd_sd_lt_0.2w": s["vdd_in_sd_w"] < 0.2, "power_rate_ge_9hz": s["rate_hz"] >= 9,
              "ts_monotonic": s["ts_monotonic"]}
    procs, n_it = top_procs(a.top)
    busy = {k: round(v, 2) for k, v in procs.items() if v > 1.0 and not k.startswith(SAMPLERS)}
    checks["no_foreign_proc_gt_1pct"] = not busy
    olines = [l for l in open(a.ollama_ps).read().splitlines() if l.strip()]
    checks["ollama_ps_empty"] = len(olines) <= 1
    bad_listen = []
    for l in open(a.ss).read().splitlines()[1:]:
        f = l.split()
        if len(f) < 4:
            continue
        addr = f[3]
        port = addr.rsplit(":", 1)[-1]
        if port == "11434":
            if not addr.startswith("127.0.0.1"):
                bad_listen.append(l)
        elif port not in ALLOWED_PORTS:
            bad_listen.append(l)
    checks["listeners_allowed_only"] = not bad_listen
    swap_used = [int(l.split()[2]) for l in open(a.free) if l.startswith("Swap")][0]
    checks["swap_used_lt_50mb"] = swap_used < 50
    ok = all(checks.values())
    print(json.dumps({"pass": ok, "checks": checks, "summary": s, "top_iterations": n_it,
                      "procs_over_1pct_excluding_samplers": busy,
                      "top_procs_avg_pct": dict(sorted(((k, round(v, 2)) for k, v in procs.items()), key=lambda x: -x[1])[:15]),
                      "bad_listeners": bad_listen, "swap_used_mb": swap_used, "ollama_ps": olines}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
