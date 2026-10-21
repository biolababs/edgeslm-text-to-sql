#!/usr/bin/env python3
"""LoadGen SingleStream run against a model served by Ollama, with on-board power (PLAN 6.3).

Follows the MLPerf LoadGen SingleStream scenario; it is NOT an MLPerf result (models are not
reference models, no accuracy mode or compliance tests, power from on-board INA3221 VDD_IN).

Revised from prior-code/loadgen_bench.py:
  - the response's model is asserted equal to the requested tag (the exploratory bug);
  - QSL = pre-rendered request files (identical prompts to the accuracy runs);
  - every sample is logged: index, rid, t_start_ns, t_end_ns, eval_count, raw output;
  - power_log.py (tegrastats 100 ms, ns timestamps) runs alongside; energy is integrated
    (trapezoid) over [first issue, last completion] and compared with mean power x duration;
  - seeds and settings are written to user.conf in the output folder.

    python loadgen_bench.py --model qc15-q4-ollama --requests req/nyc0.jsonl req/cur0.jsonl \
        --gen config/generation.json --out perf/qc15 --min-queries 1024 --min-duration-s 600 \
        --expected-latency-s 2.5 --idle-w 6.6
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import mlperf_loadgen as lg

from run_ollama import chat

SEEDS = {"qsl_rng_seed": 2026101601, "sample_index_rng_seed": 2026101602, "schedule_rng_seed": 2026101603}
VDD = re.compile(r"VDD_IN (\d+)mW")


def read_power(path):
    pts = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            continue
        ns, _, rest = line.partition(" ")
        m = VDD.search(rest)
        if m:
            pts.append((int(ns), int(m.group(1)) / 1000.0))
    return pts


def integrate(pts, t0, t1):
    """Trapezoidal integral of power (W) over [t0, t1] ns, linear interpolation at the edges. Joules."""
    if t1 <= t0 or len(pts) < 2:
        return float("nan")

    def p_at(t):
        for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                return pa + (pb - pa) * (t - ta) / (tb - ta) if tb > ta else pa
        return pts[0][1] if t < pts[0][0] else pts[-1][1]

    inner = [(t, p) for t, p in pts if t0 < t < t1]
    seq = [(t0, p_at(t0))] + inner + [(t1, p_at(t1))]
    return sum((tb - ta) / 1e9 * (pa + pb) / 2 for (ta, pa), (tb, pb) in zip(seq, seq[1:]))


def parse_summary(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", type=Path, nargs="+", required=True)
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-queries", type=int, default=1024)
    ap.add_argument("--min-duration-s", type=int, default=600)
    ap.add_argument("--expected-latency-s", type=float, required=True)
    ap.add_argument("--idle-w", type=float, help="board-idle median VDD_IN for above-idle energy")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--power-log", default=str(Path(__file__).with_name("power_log.py")))
    args = ap.parse_args()

    gen = json.loads(args.gen.read_text(encoding="utf-8"))
    reqs = []
    for p in args.requests:
        with open(p, encoding="utf-8") as f:
            reqs += [json.loads(l) for l in f if l.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    for q in reqs[:3]:  # warm-up (untimed): weights resident, caches warm
        r = chat(args.host, args.model, q["messages"], gen)
        assert r.get("model") == args.model, (r.get("model"), args.model)

    samples_log = open(args.out / "samples.jsonl", "w", encoding="utf-8")
    state = {"n": 0}

    def issue_queries(qs):
        for s in qs:
            q = reqs[s.index]
            t0 = time.time_ns()
            r = chat(args.host, args.model, q["messages"], gen)
            t1 = time.time_ns()
            if r.get("model") != args.model:
                sys.exit(f"server answered with model {r.get('model')!r}, requested {args.model!r}")
            samples_log.write(json.dumps({"seq": state["n"], "index": s.index, "rid": q["rid"], "t_start_ns": t0,
                                          "t_end_ns": t1, "eval_count": r.get("eval_count"),
                                          "prompt_eval_count": r.get("prompt_eval_count"),
                                          "prompt_eval_duration": r.get("prompt_eval_duration"),
                                          "eval_duration": r.get("eval_duration"),
                                          "done_reason": r.get("done_reason"), "model": r.get("model"),
                                          "raw": r["message"]["content"]}, ensure_ascii=False) + "\n")
            state["n"] += 1
            if state["n"] == 1 or state["n"] % 25 == 0:
                print(f"PROGRESS {state['n']}/{args.min_queries} loadgen {args.model} {(t1 - t0) / 1e9:.2f}s", flush=True)
            lg.QuerySamplesComplete([lg.QuerySampleResponse(s.id, 0, 0)])

    sut = lg.ConstructSUT(issue_queries, lambda: None)
    qsl = lg.ConstructQSL(len(reqs), len(reqs), lambda idx: None, lambda idx: None)
    settings = lg.TestSettings()
    settings.scenario = lg.TestScenario.SingleStream
    settings.mode = lg.TestMode.PerformanceOnly
    settings.min_query_count = args.min_queries
    settings.min_duration_ms = args.min_duration_s * 1000
    settings.single_stream_expected_latency_ns = int(args.expected_latency_s * 1e9)
    for k, v in SEEDS.items():
        setattr(settings, k, v)
    conf = {"scenario": "SingleStream", "mode": "PerformanceOnly", "min_query_count": args.min_queries,
            "min_duration_ms": args.min_duration_s * 1000,
            "single_stream_expected_latency_ns": int(args.expected_latency_s * 1e9),
            "target_latency_percentile": 90, "performance_sample_count": len(reqs), **SEEDS,
            "readback": {k: getattr(settings, k) for k in SEEDS}}
    (args.out / "user.conf").write_text("\n".join(f"*.SingleStream.{k} = {v}" for k, v in conf.items()) + "\n")
    log_settings = lg.LogSettings()
    log_settings.log_output.outdir = str(args.out)
    log_settings.log_output.copy_summary_to_stdout = False

    plog = args.out / "power.log"
    pw = subprocess.Popen([sys.executable, args.power_log, "--out", str(plog), "--interval", "100"])
    time.sleep(2)
    t_wall0 = time.time_ns()
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    t_wall1 = time.time_ns()
    time.sleep(1)
    pw.terminate(); pw.wait()
    samples_log.close()
    lg.DestroyQSL(qsl); lg.DestroySUT(sut)

    smp = [json.loads(l) for l in open(args.out / "samples.jsonl", encoding="utf-8")]
    t_first, t_last = smp[0]["t_start_ns"], smp[-1]["t_end_ns"]
    pts = read_power(plog)
    e_total = integrate(pts, t_first, t_last)
    dur = (t_last - t_first) / 1e9
    in_win = [p for t, p in pts if t_first <= t <= t_last]
    mean_w = sum(in_win) / len(in_win) if in_win else float("nan")
    ts = [t for t, _ in pts]
    rate = (len(ts) - 1) / ((ts[-1] - ts[0]) / 1e9) if len(ts) > 1 else float("nan")
    s = parse_summary(args.out / "mlperf_log_summary.txt")
    g = lambda k: int(s[k]) / 1e9 if k in s else None
    n = len(smp)
    res = {"model": args.model, "models_in_responses": sorted({x["model"] for x in smp}), "queries": n,
           "duration_s": dur, "qps": n / dur if dur else None,
           "p50_latency_s": g("50.00 percentile latency (ns)"), "p90_latency_s": g("90.00 percentile latency (ns)"),
           "p99_latency_s": g("99.00 percentile latency (ns)"), "mean_latency_s": g("Mean latency (ns)"),
           "early_stopping_p90_s": g("Early stopping 90th percentile estimate"),
           "loadgen_result": s.get("Result is"), "mean_gen_tokens": sum((x["eval_count"] or 0) for x in smp) / max(n, 1),
           "cap_hits": sum(x["done_reason"] == "length" for x in smp),
           "power_samples": len(pts), "power_rate_hz": rate, "power_ts_monotonic": all(b > a for a, b in zip(ts, ts[1:])),
           "energy_total_j": e_total, "energy_per_query_j": e_total / n if n else None,
           "mean_power_w": mean_w, "energy_meanpower_x_dur_j": mean_w * dur,
           "integration_vs_meanpower_rel_diff": (e_total - mean_w * dur) / (mean_w * dur) if mean_w == mean_w else None,
           "idle_w": args.idle_w,
           "energy_above_idle_per_query_j": (e_total - args.idle_w * dur) / n if args.idle_w and n else None,
           "loadgen_wall_s": (t_wall1 - t_wall0) / 1e9}
    (args.out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    if res["models_in_responses"] != [args.model]:
        sys.exit("model mismatch in responses")


if __name__ == "__main__":
    main()
