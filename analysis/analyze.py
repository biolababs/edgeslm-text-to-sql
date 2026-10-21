"""Session 1 analysis: accuracy, prompting effects, energy, and deployment tables.

Reads build/scores/<run>/summary.json (produced by score.py) and the per-model
detail files. Writes build/numbers.json, build/tables/*.md, build/figures/*.png.

    python analysis/analyze.py --run c9-main-20261017-s1
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from common import number, proportion_ci, save_table, write_numbers, BUILD, TABLES  # noqa: E402

MODELS_ORDER = [
    "prem1b-q4-ours",
    "qc15-q4-ollama",
    "qc3-q4-ollama",
    "xiyan3b-q4-ours",
    "llama32-3b-q4-ollama",
    "qwen3-4b-q4-ollama",
]
LABELS = {
    "prem1b-q4-ours": "Prem-1B-SQL",
    "qc15-q4-ollama": "QwenCoder-1.5B",
    "qc3-q4-ollama": "QwenCoder-3B",
    "xiyan3b-q4-ours": "XiYanSQL-3B",
    "llama32-3b-q4-ollama": "Llama-3.2-3B",
    "qwen3-4b-q4-ollama": "Qwen3-4B",
}
PARAMS = {
    "prem1b-q4-ours": "1.35B",
    "qc15-q4-ollama": "1.5B",
    "qc3-q4-ollama": "3.1B",
    "xiyan3b-q4-ours": "3.1B",
    "llama32-3b-q4-ollama": "3.2B",
    "qwen3-4b-q4-ollama": "4.0B",
}
MODEL_TYPE = {
    "prem1b-q4-ours": "SQL",
    "qc15-q4-ollama": "Code",
    "qc3-q4-ollama": "Code",
    "xiyan3b-q4-ours": "SQL",
    "llama32-3b-q4-ollama": "General",
    "qwen3-4b-q4-ollama": "General",
}

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class SimpleDF:
    """Minimal DataFrame replacement when pandas is unavailable."""
    def __init__(self, data, columns):
        self.columns = columns
        self._rows = data

    def itertuples(self, index=False):
        for row in self._rows:
            yield tuple(row[c] for c in self.columns)

    def to_csv(self, path, index=False):
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(self.columns) + "\n")
            for row in self._rows:
                f.write(",".join(str(row.get(c, "")) for c in self.columns) + "\n")

    def to_markdown(self, index=False, floatfmt=".2f"):
        def fmt(v):
            return format(v, floatfmt) if isinstance(v, float) else str(v)
        header = "| " + " | ".join(self.columns) + " |"
        sep = "| " + " | ".join("---" for _ in self.columns) + " |"
        rows = []
        for row in self._rows:
            rows.append("| " + " | ".join(fmt(row.get(c, "")) for c in self.columns) + " |")
        return "\n".join([header, sep, *rows])


def make_df(data, columns):
    if HAS_PANDAS:
        return pd.DataFrame(data, columns=columns)
    return SimpleDF(data, columns)


def pct(x):
    return round(100 * x, 1) if x is not None and not (isinstance(x, float) and math.isnan(x)) else None


def ci_str(correct, total):
    lo, hi = proportion_ci(correct, total)
    return f"[{pct(lo)}, {pct(hi)}]"


def load_details(scores_dir, model, cond):
    p = scores_dir / model / f"{cond}.details.jsonl"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="c9-main-20261017-s1")
    a = ap.parse_args()

    scores_dir = BUILD / "scores" / a.run
    summary_path = scores_dir / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run score.py first.")
        sys.exit(1)

    S = json.loads(summary_path.read_text(encoding="utf-8"))

    # ── Table 1: Main accuracy (RQ1, RQ2, RQ4) ──
    rows = []
    for m in MODELS_ORDER:
        r = {"Model": LABELS[m], "Params": PARAMS[m], "Type": MODEL_TYPE[m]}

        # NYC zero-shot
        k0 = f"{m}/nyc0"
        if k0 in S:
            ex0 = S[k0]["strict"]
            n0 = S[k0]["n"]
            r["NYC 0-shot"] = pct(ex0)
            r["NYC 0-shot CI"] = ci_str(int(round(ex0 * n0)), n0)
            number(f"{m}_nyc0_ex", pct(ex0), 1)

        # NYC 3-shot S-sim
        ks = f"{m}/nycsim"
        if ks in S:
            exs = S[ks]["strict"]
            ns = S[ks]["n"]
            r["NYC S-sim"] = pct(exs)
            r["NYC S-sim CI"] = ci_str(int(round(exs * ns)), ns)
            number(f"{m}_nycsim_ex", pct(exs), 1)
            if "strict_seen" in S[ks]:
                r["S-sim seen"] = pct(S[ks]["strict_seen"])
                r["S-sim unseen"] = pct(S[ks].get("strict_unseen"))
                number(f"{m}_nycsim_seen", pct(S[ks]["strict_seen"]), 1)
                number(f"{m}_nycsim_unseen", pct(S[ks].get("strict_unseen")), 1)
            else:
                # Exposure is recorded in the request id, not the item name:
                # "<item>|<pool fold>|<seen|unseen>". Each of the 449 items is
                # measured twice, once with its own template present in the
                # example pool and once with it withheld. The item name is only
                # leftover dataset naming and does not indicate exposure.
                det = load_details(scores_dir, m, "nycsim")
                unseen = [d for d in det if d["rid"].rsplit("|", 1)[-1] == "unseen"]
                seen = [d for d in det if d["rid"].rsplit("|", 1)[-1] == "seen"]
                if seen and unseen:
                    s_seen = sum(d["strict"] for d in seen) / len(seen)
                    s_unseen = sum(d["strict"] for d in unseen) / len(unseen)
                    # The two groups must partition the condition: their weighted
                    # mean has to reproduce the pooled accuracy.
                    pooled = (s_seen * len(seen) + s_unseen * len(unseen)) / len(det)
                    if abs(pooled - S[ks]["strict"]) > 1e-9 or len(seen) + len(unseen) != S[ks]["n"]:
                        sys.exit(f"ERROR: seen/unseen split does not reconcile for {m}/nycsim: "
                                 f"{len(seen)}+{len(unseen)} of {S[ks]['n']}, "
                                 f"pooled {pooled:.6f} vs reported {S[ks]['strict']:.6f}")
                    number(f"{m}_nycsim_seen_n", len(seen))
                    number(f"{m}_nycsim_unseen_n", len(unseen))
                    S[ks]["strict_seen"] = s_seen
                    S[ks]["strict_unseen"] = s_unseen
                    r["S-sim seen"] = pct(s_seen)
                    r["S-sim unseen"] = pct(s_unseen)
                    number(f"{m}_nycsim_seen", pct(s_seen), 1)
                    number(f"{m}_nycsim_unseen", pct(s_unseen), 1)

        # NYC 3-shot S-fixed
        kf = f"{m}/nycfix"
        if kf in S:
            r["NYC S-fixed"] = pct(S[kf]["strict"])
            number(f"{m}_nycfix_ex", pct(S[kf]["strict"]), 1)

        # Curated
        kc0 = f"{m}/cur0"
        if kc0 in S:
            r["Cur 0-shot"] = pct(S[kc0]["strict"])
            number(f"{m}_cur0_ex", pct(S[kc0]["strict"]), 1)
        kcs = f"{m}/cursim"
        if kcs in S:
            r["Cur S-sim"] = pct(S[kcs]["strict"])
            number(f"{m}_cursim_ex", pct(S[kcs]["strict"]), 1)
        kcf = f"{m}/curfix"
        if kcf in S:
            r["Cur S-fixed"] = pct(S[kcf]["strict"])
            number(f"{m}_curfix_ex", pct(S[kcf]["strict"]), 1)

        # Spider
        ksp = f"{m}/spider0"
        if ksp in S and "spider" in S[ksp]:
            sp = S[ksp]["spider"]
            r["Spider EX"] = pct(sp.get("ex"))
            r["Spider TS"] = pct(sp.get("ts"))
            r["Spider EM"] = pct(sp.get("em"))
            number(f"{m}_spider_ex", pct(sp.get("ex")), 1)
            if sp.get("ts") is not None:
                number(f"{m}_spider_ts", pct(sp.get("ts")), 1)
            if sp.get("em") is not None:
                number(f"{m}_spider_em", pct(sp.get("em")), 1)

        # BIRD (evaluator returns percentages, not fractions)
        kb = f"{m}/bird0"
        if kb in S and "bird_v1" in S[kb]:
            bv = S[kb]["bird_v1"]
            bird_ex = round(bv["ex"], 1) if bv.get("ex") is not None else None
            # The evaluator reports soft F1 under "f1", already a percentage.
            bird_f1 = round(bv["f1"], 1) if bv.get("f1") is not None else None
            r["BIRD EX"] = bird_ex
            r["BIRD F1"] = bird_f1
            number(f"{m}_bird_ex", bird_ex, 1)
            if bird_f1 is not None:
                number(f"{m}_bird_f1", bird_f1, 1)
            # Annotation sensitivity must be read on a fixed item set. The
            # unchanged-input subset (n=369) is easier than the full 500, so
            # comparing the corrected subset against the full v1 score measures
            # the change of subset, not the change of reference SQL.
            arc = S[kb].get("bird_arcwise_unchanged")
            base = S[kb].get("bird_v1_on_arcwise_unchanged")
            if arc and base and arc.get("ex") is not None and base.get("ex") is not None:
                if arc.get("n") != base.get("n"):
                    sys.exit(f"ERROR: BIRD paired subsets differ for {m}: "
                             f"{base.get('n')} vs {arc.get('n')}")
                number(f"{m}_bird_paired_n", arc["n"])
                number(f"{m}_bird_paired_v1", round(base["ex"], 1), 1)
                number(f"{m}_bird_paired_corr", round(arc["ex"], 1), 1)
                number(f"{m}_bird_paired_delta", round(arc["ex"] - base["ex"], 1), 1)
                r["BIRD-corr EX"] = round(arc["ex"], 1)
                number(f"{m}_bird_corr_ex", round(arc["ex"], 1), 1)
            # The official evaluator prints per-difficulty scores; capture them
            # so the benchmark result is not a single pooled figure. All 500
            # items are scored: a query exceeding the 30 s deadline counts as
            # incorrect rather than being excluded.
            for metric in ("ex", "f1"):
                tail = bv.get(f"{metric}_raw_tail") or []
                for ln in tail:
                    parts = ln.split()
                    if len(parts) >= 5 and parts[0].upper() in ("EX", "SOFT-F1"):
                        try:
                            vals = [float(x) for x in parts[1:5]]
                        except ValueError:
                            continue
                        for lbl, v in zip(("simple", "moderate", "challenging"), vals):
                            number(f"{m}_bird_{metric}_{lbl}", round(v, 1), 1)
                        break
            full = S[kb].get("bird_arcwise_full")
            if full and full.get("ex") is not None:
                number(f"{m}_bird_full_n", full["n"])
                number(f"{m}_bird_full_ex", round(full["ex"], 1), 1)

        rows.append(r)

    cols = ["Model", "Params", "Type",
            "NYC 0-shot", "NYC S-sim", "NYC S-fixed",
            "Cur 0-shot", "Cur S-sim", "Cur S-fixed",
            "Spider EX", "Spider TS", "Spider EM", "BIRD EX", "BIRD F1"]
    df = make_df(rows, cols)
    save_table(df, "accuracy_main", floatfmt=".1f")
    print("== Table: accuracy_main ==")
    print(df.to_markdown(index=False, floatfmt=".1f"))

    # ── Table 2: Template generalisation (RQ1, RQ3) ──
    gen_rows = []
    for m in MODELS_ORDER:
        ks = f"{m}/nycsim"
        if ks not in S:
            continue
        d = S[ks]
        seen = d.get("strict_seen")
        unseen = d.get("strict_unseen")
        if seen is None or unseen is None:
            continue
        gap = pct(seen) - pct(unseen) if seen is not None and unseen is not None else None
        gen_rows.append({
            "Model": LABELS[m],
            "S-sim overall": pct(d["strict"]),
            "Seen": pct(seen),
            "Unseen": pct(unseen),
            "Gap (pp)": round(gap, 1) if gap is not None else None,
        })
        number(f"{m}_gen_gap", round(gap, 1) if gap is not None else None, 1)

    if gen_rows:
        gcols = ["Model", "S-sim overall", "Seen", "Unseen", "Gap (pp)"]
        gdf = make_df(gen_rows, gcols)
        save_table(gdf, "template_generalisation", floatfmt=".1f")
        print("\n== Table: template_generalisation ==")
        print(gdf.to_markdown(index=False, floatfmt=".1f"))

    # ── Table: BIRD annotation sensitivity, measured on a fixed item set ──
    bird_rows = []
    for m in MODELS_ORDER:
        kb = f"{m}/bird0"
        if kb not in S:
            continue
        d = S[kb]
        arc, base = d.get("bird_arcwise_unchanged"), d.get("bird_v1_on_arcwise_unchanged")
        full, v1 = d.get("bird_arcwise_full"), d.get("bird_v1")
        if not (arc and base and v1):
            continue
        bird_rows.append({
            "Model": LABELS[m],
            "v1 full (n=500)": round(v1["ex"], 1),
            "v1 on subset": round(base["ex"], 1),
            "Corrected on subset": round(arc["ex"], 1),
            "Delta (pp)": round(arc["ex"] - base["ex"], 1),
        })
    if bird_rows:
        # The 498-item corrected set is deliberately NOT a column here. It is a
        # different item set from the 500-item v1 set and mixes two question
        # wordings, so placing the two side by side would re-create the
        # denominator artifact this table exists to expose. It is registered as
        # a number for prose that can state those caveats.
        bcols = ["Model", "v1 full (n=500)", "v1 on subset", "Corrected on subset",
                 "Delta (pp)"]
        bdf = make_df(bird_rows, bcols)
        save_table(bdf, "bird_sensitivity", floatfmt=".1f")
        print("\n== Table: bird_sensitivity ==")
        print(bdf.to_markdown(index=False, floatfmt=".1f"))

    # ── Table 3: Latency and energy (RQ5) ──
    cost_rows = []
    for m in MODELS_ORDER:
        r = {"Model": LABELS[m]}
        for cond in ("nyc0", "nycsim", "cur0", "cursim"):
            k = f"{m}/{cond}"
            if k not in S:
                continue
            d = S[k]
            r[f"{cond} wall_s"] = round(d.get("wall_s_mean", 0), 2)
            if d.get("energy_j_mean") is not None:
                r[f"{cond} J/q"] = round(d["energy_j_mean"], 1)
            if d.get("energy_above_idle_j_mean") is not None:
                r[f"{cond} J/q above idle"] = round(d["energy_above_idle_j_mean"], 1)
                number(f"{m}_{cond}_j_above_idle", round(d["energy_above_idle_j_mean"], 1), 1)
            if d.get("energy_j_mean") is not None:
                number(f"{m}_{cond}_j_per_query", round(d["energy_j_mean"], 1), 1)
            # Energy per correct answer
            ex = d.get("strict")
            ej = d.get("energy_j_mean")
            n = d.get("n", 0)
            if ex is not None and ej is not None and ex > 0 and n > 0:
                correct = int(round(ex * n))
                total_j = ej * n
                j_per_correct = total_j / correct if correct > 0 else None
                r[f"{cond} J/correct"] = round(j_per_correct, 1) if j_per_correct else None
                number(f"{m}_{cond}_j_per_correct", round(j_per_correct, 1) if j_per_correct else None)

        # Decode speed
        k0 = f"{m}/nyc0"
        if k0 in S:
            r["decode tok/s"] = round(S[k0].get("decode_tok_s_mean", 0), 1)
            number(f"{m}_decode_toks", round(S[k0].get("decode_tok_s_mean", 0), 1))
            # Prompt-token rate is NOT prefill throughput. Every prompt shares a
            # long schema prefix that the server caches, so the reported token
            # count includes cached tokens that were never recomputed. The
            # implied compute exceeds the board's peak by more than an order of
            # magnitude, so it is kept for the record but not reported as a
            # hardware figure.
            number(f"{m}_prompt_toks_cachewarm", round(S[k0].get("prefill_tok_s_mean", 0), 1))

        cost_rows.append(r)

    ccols = ["Model", "nyc0 wall_s", "nycsim wall_s", "nyc0 J/q", "nycsim J/q",
             "nyc0 J/correct", "nycsim J/correct", "decode tok/s"]
    cdf = make_df(cost_rows, ccols)
    save_table(cdf, "latency_energy", floatfmt=".1f")
    print("\n== Table: latency_energy ==")
    print(cdf.to_markdown(index=False, floatfmt=".1f"))

    # ── Table 4: XiYan native prompt comparison (RQ4) ──
    xiyan_rows = []
    for cond_pair in [("nyc0", "nycnat"), ("cur0", "curnat")]:
        std, nat = cond_pair
        ks = f"xiyan3b-q4-ours/{std}"
        kn = f"xiyan3b-q4-ours/{nat}"
        if ks in S and kn in S:
            xiyan_rows.append({
                "Condition": f"{std} vs {nat}",
                "Common prompt": pct(S[ks]["strict"]),
                "Native prompt": pct(S[kn]["strict"]),
                "Diff (pp)": round(pct(S[kn]["strict"]) - pct(S[ks]["strict"]), 1),
            })
    if xiyan_rows:
        xdf = make_df(xiyan_rows, ["Condition", "Common prompt", "Native prompt", "Diff (pp)"])
        save_table(xdf, "xiyan_native", floatfmt=".1f")
        print("\n== Table: xiyan_native ==")
        print(xdf.to_markdown(index=False, floatfmt=".1f"))

    # ── Table 5: Error taxonomy ──
    tax_rows = []
    for m in MODELS_ORDER:
        k0 = f"{m}/nyc0"
        if k0 not in S:
            continue
        classes = S[k0].get("classes", {})
        n = S[k0]["n"]
        tax_rows.append({
            "Model": LABELS[m],
            "correct": classes.get("correct", 0),
            "wrong_result": classes.get("wrong_result", 0),
            "exec_error": classes.get("exec_error", 0),
            "no_sql": classes.get("no_sql", 0),
            "truncated": classes.get("truncated", 0),
        })
    if tax_rows:
        tcols = ["Model", "correct", "wrong_result", "exec_error", "no_sql", "truncated"]
        tdf = make_df(tax_rows, tcols)
        save_table(tdf, "error_taxonomy", floatfmt=".0f")
        print("\n== Table: error_taxonomy ==")
        print(tdf.to_markdown(index=False, floatfmt=".0f"))

    # ── Summary numbers ──
    # Best models
    best_nycsim = max(MODELS_ORDER, key=lambda m: S.get(f"{m}/nycsim", {}).get("strict", 0))
    number("best_nycsim_model", LABELS[best_nycsim])
    number("best_nycsim_ex", pct(S[f"{best_nycsim}/nycsim"]["strict"]), 1)

    best_spider = max(MODELS_ORDER, key=lambda m: S.get(f"{m}/spider0", {}).get("spider", {}).get("ex", 0))
    sp_ex = S.get(f"{best_spider}/spider0", {}).get("spider", {}).get("ex")
    number("best_spider_model", LABELS[best_spider])
    number("best_spider_ex", pct(sp_ex), 1)

    best_bird = max(MODELS_ORDER, key=lambda m: S.get(f"{m}/bird0", {}).get("bird_v1", {}).get("ex", 0))
    bd_ex = S.get(f"{best_bird}/bird0", {}).get("bird_v1", {}).get("ex")
    number("best_bird_model", LABELS[best_bird])
    number("best_bird_ex", round(bd_ex, 1) if bd_ex is not None else None, 1)

    # 3-shot lift
    for m in MODELS_ORDER:
        z = S.get(f"{m}/nyc0", {}).get("strict", 0)
        s = S.get(f"{m}/nycsim", {}).get("strict", 0)
        number(f"{m}_3shot_lift", round(pct(s) - pct(z), 1), 1)

    # Decompose the three-shot gain. Moving from zero-shot to a prompt whose
    # examples exclude the item's own template isolates what examples supply
    # irrespective of template match; the remaining step to a covered pool is
    # what template exposure adds.
    shares, pooled_shares = [], []
    for m in MODELS_ORDER:
        d = S.get(f"{m}/nycsim", {})
        z = S.get(f"{m}/nyc0", {}).get("strict")
        u, sn = d.get("strict_unseen"), d.get("strict_seen")
        if z is None or u is None or sn is None or sn <= z:
            continue
        ex_eff, tpl_eff = pct(u) - pct(z), pct(sn) - pct(u)
        share = 100 * (sn - u) / (sn - z)
        shares.append(share)
        number(f"{m}_example_effect", round(ex_eff, 1), 1)
        number(f"{m}_template_effect", round(tpl_eff, 1), 1)
        number(f"{m}_template_share", round(share, 1), 1)
        # The pooled lift mixes the two exposures in equal numbers because the
        # design fixed that rate at one half. Nothing in deployment does, so
        # the two conditional lifts are reported alongside it.
        number(f"{m}_lift_covered", round(pct(sn) - pct(z), 1), 1)
        number(f"{m}_lift_withheld", round(pct(u) - pct(z), 1), 1)
        # The headline S-similar figure pools the two exposures in equal
        # numbers, so exposure contributes only half the gap against that
        # denominator. The two shares answer different questions and must not
        # be quoted interchangeably.
        pooled = S.get(f"{m}/nycsim", {}).get("strict")
        if pooled is not None and pooled > z:
            share_pooled = 100 * ((sn - u) / 2) / (pooled - z)
            pooled_shares.append(share_pooled)
            number(f"{m}_template_share_pooled", round(share_pooled, 1), 1)
    # Retrieval's advantage over a fixed example set, once template coverage is
    # removed. The pooled comparison flatters retrieval because half its items
    # have their own template in the pool.
    n_better = 0
    for m in MODELS_ORDER:
        fx = S.get(f"{m}/nycfix", {}).get("strict")
        un = S.get(f"{m}/nycsim", {}).get("strict_unseen")
        if fx is None or un is None:
            continue
        adv = pct(un) - pct(fx)
        number(f"{m}_retr_adv_uncovered", round(adv, 1), 1)
        n_better += adv > 0
    number("n_retr_beats_fixed_uncovered", n_better)

    gaps = [pct(S[f"{m}/nycsim"]["strict_seen"]) - pct(S[f"{m}/nycsim"]["strict_unseen"])
            for m in MODELS_ORDER
            if S.get(f"{m}/nycsim", {}).get("strict_seen") is not None]
    if gaps:
        number("gen_gap_min", round(min(gaps), 1), 1)
        number("gen_gap_max", round(max(gaps), 1), 1)

    for field, stem in (("strict_seen", "seen_ex"), ("strict_unseen", "unseen_ex")):
        vals = [pct(S[f"{m}/nycsim"][field]) for m in MODELS_ORDER
                if S.get(f"{m}/nycsim", {}).get(field) is not None]
        if vals:
            number(f"{stem}_min", round(min(vals), 1), 1)
            number(f"{stem}_max", round(max(vals), 1), 1)

    if shares:
        number("template_share_min", round(min(shares), 1), 1)
        number("template_share_max", round(max(shares), 1), 1)
        number("template_share_mean", round(sum(shares) / len(shares), 1), 1)
    if pooled_shares:
        number("template_share_pooled_min", round(min(pooled_shares), 1), 1)
        number("template_share_pooled_max", round(max(pooled_shares), 1), 1)
        number("template_share_pooled_mean",
               round(sum(pooled_shares) / len(pooled_shares), 1), 1)

    # A condition can be dominated by one repeated mistake rather than by broad
    # incapacity. Where that happens the score needs reading differently, so
    # the largest single execution-error mode is reported per condition.
    for m in MODELS_ORDER:
        for cond in ("nyc0", "nycfix"):
            det = load_details(scores_dir, m, cond)
            errs = [d["error"] for d in det if d.get("class") == "exec_error" and d.get("error")]
            if not errs:
                continue
            msgs = Counter(e.strip() for e in errs)
            top, cnt = msgs.most_common(1)[0]
            if cnt >= 0.1 * len(det):
                number(f"{m}_{cond}_top_error_n", cnt)
                number(f"{m}_{cond}_top_error_pct", round(100 * cnt / len(det), 1), 1)

    # Model count
    number("n_models", len(MODELS_ORDER))
    number("n_nyc_eval", S.get(f"{MODELS_ORDER[0]}/nyc0", {}).get("n", 449))

    # What the exposure conditions actually put in the prompt. This describes
    # the treatment rather than inferring a mechanism from the outcome.
    req = ROOT / "experiment" / "requests" / "nycsim.jsonl"
    pool_f = ROOT / "experiment" / "data" / "nyc_pool.jsonl"
    if req.exists() and pool_f.exists():
        pool = {r["id"]: r.get("template_id")
                for r in (json.loads(l) for l in pool_f.read_text(encoding="utf-8").splitlines() if l.strip())}
        reqs = [json.loads(l) for l in req.read_text(encoding="utf-8").splitlines() if l.strip()]
        for role in ("seen", "unseen"):
            sub = [r for r in reqs if r.get("role") == role]
            if not sub:
                continue
            same = [sum(1 for s in r["shot_ids"] if pool.get(s) == r["template_id"]) for r in sub]
            sims = [s for r in sub for s in r["shot_sims"]]
            number(f"shots_{role}_n", len(sub))
            number(f"shots_{role}_all_same", sum(1 for c in same if c == 3))
            number(f"shots_{role}_none_same", sum(1 for c in same if c == 0))
            number(f"shots_{role}_sim_mean", round(sum(sims) / len(sims), 3), 3)
        # Fold balance, so a reader can check no single rotation dominates.
        from collections import Counter as _C
        fc = _C(r.get("pool_fold") for r in reqs if r.get("pool_fold"))
        for fold, cnt in sorted(fc.items()):
            number(f"requests_pool_{fold}", cnt)
        if fc:
            number("requests_pool_fold_min", min(fc.values()))
            number("requests_pool_fold_max", max(fc.values()))

    import stats_tests
    stats_tests.make_tables(scores_dir, MODELS_ORDER, LABELS, make_df)

    import figures
    figures.make_all(S, MODELS_ORDER, LABELS, MODEL_TYPE, PARAMS)

    write_numbers()
    print(f"\nWrote {len(json.loads((BUILD / 'numbers.json').read_text()))} numbers to build/numbers.json")
    print("ANALYSIS_DONE")


if __name__ == "__main__":
    main()
