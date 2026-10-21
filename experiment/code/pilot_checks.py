#!/usr/bin/env python3
"""Pilot gate checks (PLAN 10 items 1-10) computed from a copied-back pilot run folder, plus the
updated full-run time estimate (PLAN 11). Writes <out>/pilot_checks.json. Derived; re-creatable.

    python pilot_checks.py --run results/raw/pilot-20261016 --scores build/scores/pilot-20261016 --out build/pilot
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path

MODELS = ["prem1b-q4-ours", "qc15-q4-ollama", "qc3-q4-ollama", "xiyan3b-q4-ours", "llama32-3b-q4-ollama",
          "qwen3-4b-q4-ollama", "qc15-q4-ours", "qc15-q8-ours", "qc15-f16-ours"]
VDD = re.compile(r"VDD_IN (\d+)mW"); SWAP = re.compile(r"SWAP (\d+)/"); RAM = re.compile(r"RAM (\d+)/")
TJ = re.compile(r"tj@([\d.]+)C")


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def tegra(paths):
    rows = []
    for p in paths:
        for line in open(p):
            if line.startswith("#"):
                continue
            ns, _, rest = line.partition(" ")
            v = VDD.search(rest)
            if v:
                rows.append((int(ns), int(v.group(1)) / 1000, int(SWAP.search(rest).group(1)), int(RAM.search(rest).group(1)),
                             float(TJ.search(rest).group(1))))
    return sorted(rows)


def window(rows, t0, t1):
    return [r for r in rows if t0 <= r[0] <= t1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True); ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); run = a.run
    ev = []
    for line in open(run / "logs" / "events.tsv"):
        ns, txt = line.rstrip("\n").split("\t", 1); ev.append((int(ns), txt))
    evt = lambda s: [t for t, x in ev if x.startswith(s)]
    rows = tegra(sorted((run / "power").glob("session*.log")))
    res = {}

    # power log sanity (item 8)
    ts = [r[0] for r in rows]
    gaps = [(b - a) / 1e9 for a, b in zip(ts, ts[1:])]
    res["power_log"] = {"samples": len(rows), "rate_hz_median": 1 / st.median(gaps) if gaps else None,
                        "max_gap_s": max(gaps) if gaps else None, "monotonic": all(g > 0 for g in gaps)}
    gpu = []
    for p in sorted((run / "power").glob("gpu_*.tsv")):
        for line in open(p):
            f = line.rstrip("\n").split("\t")
            if f[0].isdigit():
                gpu.append(f)
    res["gpu_sampler"] = {"samples": len(gpu), "gpu_hz_min": min(int(f[1]) for f in gpu) if gpu else None,
                          "any_crit_alarm": any("1" in f[5].split(",") for f in gpu) if gpu else None,
                          "tj_max_c": max(int(f[4]) for f in gpu) / 1000 if gpu else None}

    # LoRA training (item 6)
    tl = json.loads((run / "train" / "f1_smoke" / "train_log.json").read_text())
    t0, t1 = evt("lora_train start")[0], evt("lora_train end")[0]
    w = window(rows, t0, t1)
    res["lora_train"] = {"steps": tl["steps"], "examples": tl["examples_seen"], "train_s": tl["train_seconds"],
                         "s_per_step": tl["s_per_step"], "s_per_example": tl["train_seconds"] / tl["examples_seen"],
                         "peak_gpu_mem_gb": tl["peak_mem_gb"], "loss_first_last": [tl["log"][0]["loss"], tl["log"][-1]["loss"]],
                         "wall_incl_load_s": (t1 - t0) / 1e9,
                         "swap_delta_mb": (w[-1][2] - w[0][2]) if w else None, "swap_max_mb": max(r[2] for r in w) if w else None,
                         "ram_peak_mb": max(r[3] for r in w) if w else None, "tj_max_c": max(r[4] for r in w) if w else None,
                         "energy_kj": sum((b[0] - a[0]) / 1e9 * (a[1] + b[1]) / 2 for a, b in zip(w, w[1:])) / 1000 if w else None,
                         "mean_vdd_w": st.mean(r[1] for r in w) if w else None}

    # per model: loaded idle, ollama ps, sha, template check, run timings
    sc = json.loads((a.scores / "summary.json").read_text())
    per = {}
    for m in MODELS:
        d = {}
        li = evt(f"loaded_idle start {m}"), evt(f"loaded_idle end {m}")
        if li[0] and li[1]:
            ww = window(rows, li[0][0] + 10e9, li[1][0])
            d["loaded_idle_w_median"] = st.median(r[1] for r in ww) if ww else None
            d["loaded_idle_ram_mb"] = st.median(r[3] for r in ww) if ww else None
        ps = (run / "env" / f"ollama_ps_{m}.txt").read_text() if (run / "env" / f"ollama_ps_{m}.txt").exists() else ""
        d["ollama_ps"] = ps.strip().splitlines()[-1] if ps.strip() else None
        d["gpu_100pct"] = "100% GPU" in ps
        d["sha_ok"] = any(x == f"sha ok {m}" for _, x in ev)
        tc = run / "checks" / f"template_check_{m}.jsonl"
        if tc.exists():
            t = rd(tc)
            d["template_tokens_match_hf"] = sum(x["raw_prompt_eval_count"] == x["hf_n_tokens"] == x["chat_prompt_eval_count"] for x in t)
            d["template_same_output_raw_vs_chat"] = sum(x["same_output"] for x in t)
            d["template_n"] = len(t)
        conds = {k.split("/", 1)[1]: v for k, v in sc.items() if k.startswith(m + "/") and not k.endswith("_warmup")}
        n_all = sum(v["n"] for c, v in conds.items() if not c.startswith(("determinism", "lora")))
        d["n_items"] = n_all
        d["sql_extracted_rate"] = sum(v["sql_extracted"] for c, v in conds.items() if not c.startswith(("determinism", "lora"))) / n_all if n_all else None
        d["cap_hit_rate"] = sum(v["cap_hits"] for c, v in conds.items() if not c.startswith(("determinism", "lora"))) / n_all if n_all else None
        d["ctx_flags"] = sum(v["ctx_flags"] for v in conds.values())
        d["wall_s_by_cond"] = {c: round(v["wall_s_mean"], 3) for c, v in conds.items()}
        d["decode_tok_s"] = {c: round(v["decode_tok_s_mean"], 1) for c, v in conds.items()}
        d["energy_j_by_cond"] = {c: round(v.get("energy_j_mean", float("nan")), 2) for c, v in conds.items()}
        per[m] = d
    res["models"] = per

    # determinism (item 5)
    d1 = {r["rid"]: r["raw"] for r in rd(run / "acc" / "qc15-q4-ollama" / "determinism_run1.jsonl")}
    d2 = {r["rid"]: r["raw"] for r in rd(run / "acc" / "qc15-q4-ollama" / "determinism_run2.jsonl")}
    a0 = {r["rid"]: r["raw"] for r in rd(run / "acc" / "qc15-q4-ollama" / "nyc0.jsonl")}
    res["determinism"] = {"n": len(d1), "identical_run1_run2": sum(d1[k] == d2.get(k) for k in d1),
                          "identical_with_earlier_nyc0_run": sum(d1[k] == a0[k] for k in d1 if k in a0),
                          "overlap_with_nyc0": sum(k in a0 for k in d1)}

    # time estimate (PLAN 11 counts; C = curated size)
    cur = len(rd(Path(json.loads(Path("C:/Users/biola/paper/projects/c9-text-to-sql/experiment/config/score_paths.json").read_text())["experiment"]) / "curated" / "curated.jsonl"))
    counts = {"nyc0": 449 + cur, "nycsim": 898 + 0, "cursim": cur, "nycfix": 449 + cur, "spider0": 1034, "bird0": 500}
    est, tot = {}, 0.0
    for m in MODELS:
        w = per[m]["wall_s_by_cond"]
        g = lambda c: w.get(c, w.get("nyc0"))
        s = (counts["nyc0"] * g("nyc0") + counts["nycsim"] * g("nycsim") + counts["cursim"] * g("nycsim")
             + counts["spider0"] * g("spider0") + counts["bird0"] * g("bird0"))
        if not m.startswith("qc15-") or m == "qc15-q4-ollama":
            s += counts["nycfix"] * g("nycfix")
        if m == "xiyan3b-q4-ours":
            s += (449 + cur) * w.get("nycnat", g("nyc0")) + 500 * w.get("birdnat", g("bird0"))
        est[m] = round(s / 3600, 2); tot += s
    res["estimate"] = {"curated_n": cur, "accuracy_hours_by_model_excl_overhead": est, "accuracy_hours_total": round(tot / 3600, 1)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "pilot_checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
