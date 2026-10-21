"""Uncertainty estimates for the taxi-database results.

Two properties of the design rule out textbook binomial intervals:

  * Items are generated from a small set of query templates, so predictions
    within a template are correlated. The resampling unit is therefore the
    template, not the item.
  * The S-similar condition evaluates each of the 449 items twice, so its 898
    rows are repeated measurements, not independent observations. Resampling
    whole templates keeps an item's repeats together.

Every interval below is a 95% percentile cluster bootstrap over templates with
a fixed seed, so the numbers are reproducible.
"""
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from common import number, save_table  # noqa: E402

N_BOOT = 10000
SEED = 20261017


def load(scores_dir, model, cond):
    p = scores_dir / model / f"{cond}.details.jsonl"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_template(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["template_id"]].append(r)
    return g


def _mean(rows, pred=None):
    sel = [r for r in rows if pred(r)] if pred else rows
    return float(np.mean([r["strict"] for r in sel])) if sel else np.nan


def cluster_boot(groups, stat, n_boot=N_BOOT, seed=SEED):
    """Resample template clusters with replacement; return (lo, hi) at 95%."""
    rng = np.random.default_rng(seed)
    keys = list(groups)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        rows = [r for i in pick for r in groups[keys[i]]]
        v = stat(rows)
        if not np.isnan(v):
            out.append(v)
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def mcnemar(a_rows, b_rows):
    """Exact McNemar on items present in both conditions (one row per item)."""
    a = {r["id"]: r["strict"] for r in a_rows}
    b = {r["id"]: r["strict"] for r in b_rows}
    both = set(a) & set(b)
    n01 = sum(1 for i in both if not a[i] and b[i])
    n10 = sum(1 for i in both if a[i] and not b[i])
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    from scipy import stats as sps
    p = float(sps.binomtest(min(n01, n10), n, 0.5).pvalue)
    return n01, n10, p


def is_unseen(r):
    """Exposure comes from the request id tail ("...|F2|seen"), not the item name."""
    return r["rid"].rsplit("|", 1)[-1] == "unseen"


def _j_per_correct(rows):
    """Total measured energy over the resampled rows divided by the number
    correct, matching the definition used for the point estimates."""
    k = sum(r["strict"] for r in rows)
    return float(sum(r["_e"] for r in rows) / k) if k else np.nan


_POWER = {}


def _power_points(run):
    """Timestamped module power samples, as score.py reads them."""
    if run in _POWER:
        return _POWER[run]
    import re
    pts = []
    for p in sorted((run / "power").glob("session*.log")):
        with open(p, errors="ignore") as fh:
            for line in fh:
                head = line.split(None, 1)[0] if line.strip() else ""
                m = re.search(r"VDD_IN\s+(\d+)", line)
                if head.isdigit() and m:
                    pts.append((int(head), int(m.group(1)) / 1000))
    pts.sort()
    _POWER[run] = pts
    return pts


def per_request_energy(scores_dir, model, cond):
    """Joules per request, recomputed from the saved timestamps and the power
    trace by the same trapezoid rule score.py uses. Returns {} when the trace
    does not cover the condition."""
    import bisect
    run = scores_dir.parent.parent.parent / "results" / "raw" / scores_dir.name
    raw = run / "acc" / model / f"{cond}.jsonl"
    if not raw.exists():
        return {}
    pts = _power_points(run)
    if not pts:
        return {}
    ts = [t for t, _ in pts]
    out = {}
    for line in raw.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t0, t1 = r.get("t_send_ns"), r.get("t_recv_ns")
        if t0 is None or t1 is None:
            continue
        i, j = bisect.bisect_left(ts, t0), bisect.bisect_right(ts, t1)
        if i == 0 or j >= len(ts):
            continue
        seq = [(t0, pts[i - 1][1])] + pts[i:j] + [(t1, pts[j - 1][1])]
        out[r["rid"]] = sum((b[0] - a[0]) / 1e9 * (a[1] + b[1]) / 2
                            for a, b in zip(seq, seq[1:]))
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        adj = min(1.0, (len(pvals) - rank) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out


def boot_p(samples):
    """Two-sided bootstrap p-value for a difference statistic."""
    a = np.asarray(samples)
    if a.size == 0:
        return 1.0
    return float(min(1.0, 2 * min((a <= 0).mean(), (a >= 0).mean())))


def cluster_boot_dist(groups, stat, n_boot=N_BOOT, seed=SEED):
    """As cluster_boot, but returns the whole replicate distribution."""
    rng = np.random.default_rng(seed)
    keys = list(groups)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        rows = [r for i in pick for r in groups[keys[i]]]
        v = stat(rows)
        if not np.isnan(v):
            out.append(v)
    return np.array(out)


def pairwise_contrasts(scores_dir, order, labels, cond="nycsim"):
    """Every model pair under one condition, paired per request and clustered
    by template. Testing only the closest pair and generalising from it would
    understate what the design resolves."""
    data = {}
    for m in order:
        rows = load(scores_dir, m, cond)
        if rows:
            data[m] = {r["rid"]: r for r in rows}
    out = []
    for a, b in itertools.combinations([m for m in order if m in data], 2):
        keys = sorted(set(data[a]) & set(data[b]))
        if not keys:
            continue
        groups = defaultdict(list)
        for k in keys:
            groups[data[a][k]["template_id"]].append(
                {"strict": data[a][k]["strict"] - data[b][k]["strict"]})
        diff = float(np.mean([data[a][k]["strict"] - data[b][k]["strict"] for k in keys]))
        dist = cluster_boot_dist(groups, _mean)
        out.append({
            "a": a, "b": b, "diff": 100 * diff, "dist": dist,
            "lo": 100 * float(np.percentile(dist, 2.5)),
            "hi": 100 * float(np.percentile(dist, 97.5)),
            "p": boot_p(dist),
        })
    # Per-comparison intervals beside family-adjusted p-values contradict each
    # other: an interval can exclude zero for a pair Holm does not separate.
    # Simultaneous Bonferroni-level intervals are reported alongside so the
    # interval and the decision agree.
    m_tests = max(1, len(out))
    for r in out:
        half = 100 * (0.05 / (2 * m_tests))
        r["slo"] = float(np.percentile(r["dist"], half))
        r["shi"] = float(np.percentile(r["dist"], 100 - half))
        del r["dist"]
    for row, adj in zip(out, holm([r["p"] for r in out])):
        row["p_holm"] = adj
    return out


def run(scores_dir, order, labels):
    rows_ci, rows_cmp = [], []
    # Counted rather than typed: how many retrieval-vs-fixed advantages under a
    # withheld template actually separate from zero.
    n_retr_adv_sep = [0]

    for m in order:
        z = load(scores_dir, m, "nyc0")
        s = load(scores_dir, m, "nycsim")
        if not z or not s:
            continue

        gs, gz = by_template(s), by_template(z)

        sim = _mean(s)
        sim_lo, sim_hi = cluster_boot(gs, _mean)

        gap = _mean(s, lambda r: not is_unseen(r)) - _mean(s, is_unseen)
        g_lo, g_hi = cluster_boot(
            gs, lambda rs: _mean(rs, lambda r: not is_unseen(r)) - _mean(rs, is_unseen))

        # The lift pairs two conditions over the same templates.
        merged = defaultdict(list)
        for t, rs in gz.items():
            merged[t].extend({**r, "_c": "z"} for r in rs)
        for t, rs in gs.items():
            merged[t].extend({**r, "_c": "s"} for r in rs)
        lift = _mean(s) - _mean(z)
        l_lo, l_hi = cluster_boot(
            merged, lambda rs: _mean(rs, lambda r: r["_c"] == "s") - _mean(rs, lambda r: r["_c"] == "z"))

        # Share of the zero-shot-to-covered-pool gain attributable to exposure.
        # Already a percentage, so it is not rescaled like the means above.
        def share_stat(rs):
            sn = _mean(rs, lambda r: r["_c"] == "s" and not is_unseen(r))
            un = _mean(rs, lambda r: r["_c"] == "s" and is_unseen(r))
            zr = _mean(rs, lambda r: r["_c"] == "z")
            if np.isnan(sn) or np.isnan(un) or np.isnan(zr) or sn <= zr:
                return np.nan
            return 100 * (sn - un) / (sn - zr)

        sh_lo, sh_hi = cluster_boot(merged, share_stat)
        number(f"{m}_template_share_ci_lo", round(sh_lo, 1), 1)
        number(f"{m}_template_share_ci_hi", round(sh_hi, 1), 1)

        # Fixed examples against zero-shot, clustered by template. The exact
        # McNemar test below pairs the two measurements of an item but still
        # treats items from one template as independent, so this interval, not
        # that p-value, is the primary inference for this contrast.
        f = load(scores_dir, m, "nycfix")
        if f:
            gf = by_template(f)
            mf = defaultdict(list)
            for t, rs in gz.items():
                mf[t].extend({**r, "_c": "z"} for r in rs)
            for t, rs in gf.items():
                mf[t].extend({**r, "_c": "f"} for r in rs)
            fx_lo, fx_hi = cluster_boot(
                mf, lambda rs: _mean(rs, lambda r: r["_c"] == "f") - _mean(rs, lambda r: r["_c"] == "z"))
            number(f"{m}_fixlift", round(100 * (_mean(f) - _mean(z)), 1), 1)
            number(f"{m}_fixlift_ci_lo", round(100 * fx_lo, 1), 1)
            number(f"{m}_fixlift_ci_hi", round(100 * fx_hi, 1), 1)

        # Retrieval against a fixed example set with the item's own template
        # withheld. Paired within items: both conditions score the same 449
        # questions once each. This contrast carries real weight in the
        # discussion, so it gets an interval rather than a point estimate.
        if f:
            fx_by = {r["id"]: r for r in f}
            un_rows = [r for r in s if is_unseen(r)]
            gra = defaultdict(list)
            for r in un_rows:
                o = fx_by.get(r["id"])
                if o is not None:
                    gra[r["template_id"]].append({"strict": r["strict"] - o["strict"]})
            if gra:
                adv = float(np.mean([x["strict"] for v in gra.values() for x in v]))
                a_lo, a_hi = cluster_boot(gra, _mean)
                number(f"{m}_retr_adv_unc_ci_lo", round(100 * a_lo, 1), 1)
                number(f"{m}_retr_adv_unc_ci_hi", round(100 * a_hi, 1), 1)
                number(f"{m}_retr_adv_unc_pt", round(100 * adv, 1), 1)
                if a_lo > 0 or a_hi < 0:
                    n_retr_adv_sep[0] += 1

        # Energy per correct answer, resampling templates so that measured
        # energy and correctness move together. This covers variation across
        # the workload only; one session cannot estimate run-to-run variation.
        ej = per_request_energy(scores_dir, m, "nycsim")
        if ej:
            ge = defaultdict(list)
            for r in s:
                if r["rid"] in ej:
                    ge[r["template_id"]].append({**r, "_e": ej[r["rid"]]})
            if ge:
                e_lo, e_hi = cluster_boot(ge, _j_per_correct)
                number(f"{m}_jpc_ci_lo", round(e_lo, 1), 1)
                number(f"{m}_jpc_ci_hi", round(e_hi, 1), 1)

        for key, val in (("nycsim_ci_lo", sim_lo), ("nycsim_ci_hi", sim_hi),
                         ("gen_gap_ci_lo", g_lo), ("gen_gap_ci_hi", g_hi),
                         ("lift_ci_lo", l_lo), ("lift_ci_hi", l_hi)):
            number(f"{m}_{key}", round(100 * val, 1), 1)

        rows_ci.append({
            "Model": labels[m],
            "S-sim %": round(100 * sim, 1),
            "95% CI": f"[{100 * sim_lo:.1f}, {100 * sim_hi:.1f}]",
            "Gap (pp)": round(100 * gap, 1),
            "Gap 95% CI": f"[{100 * g_lo:.1f}, {100 * g_hi:.1f}]",
            "Gap excludes 0": "no" if g_lo <= 0 <= g_hi else "yes",
        })

    # Exposure is paired within items: each item is measured once with its own
    # template in the pool and once with it withheld.
    for m in order:
        det = load(scores_dir, m, "nycsim")
        if not det:
            continue
        seen = [r for r in det if not is_unseen(r)]
        unseen = [r for r in det if is_unseen(r)]
        if not seen or not unseen:
            continue
        n01, n10, p = mcnemar(unseen, seen)
        number(f"{m}_exposure_p", float(f"{p:.3g}"))
        # Discordant pairs: correct only when the template was present / withheld.
        number(f"{m}_exposure_only_present", n01)
        number(f"{m}_exposure_only_withheld", n10)
        number(f"{m}_seen_ex", round(100 * _mean(seen), 1), 1)
        number(f"{m}_unseen_ex", round(100 * _mean(unseen), 1), 1)
        rows_cmp.append({
            "Comparison": f"{labels[m]}: template withheld vs present",
            "Only A correct": n10, "Only B correct": n01,
            "p (exact McNemar)": f"{p:.3g}",
        })

    number("n_retr_adv_unc_separated", n_retr_adv_sep[0])

    # The curated set carries no template structure - its items were drafted
    # individually and carry no template_id - so each item is its own cluster
    # and this is an ordinary nonparametric bootstrap. A small sample is a
    # reason to quantify uncertainty rather than to omit it.
    for m in order:
        for cond in ("cur0", "cursim", "curfix"):
            rows = load(scores_dir, m, cond)
            if not rows:
                continue
            groups = {r["id"]: [r] for r in rows}
            lo, hi = cluster_boot(groups, _mean)
            number(f"{m}_{cond}_ci_lo", round(100 * lo, 1), 1)
            number(f"{m}_{cond}_ci_hi", round(100 * hi, 1), 1)
        z, s = load(scores_dir, m, "cur0"), load(scores_dir, m, "cursim")
        if z and s:
            zi = {r["id"]: r["strict"] for r in z}
            paired = {r["id"]: [{"strict": r["strict"] - zi[r["id"]]}]
                      for r in s if r["id"] in zi}
            if paired:
                d = float(np.mean([x["strict"] for v in paired.values() for x in v]))
                lo, hi = cluster_boot(paired, _mean)
                number(f"{m}_cur_lift", round(100 * d, 1), 1)
                number(f"{m}_cur_lift_ci_lo", round(100 * lo, 1), 1)
                number(f"{m}_cur_lift_ci_hi", round(100 * hi, 1), 1)

    # Zero-shot against fixed examples is genuinely paired: one row per item.
    for m in order:
        z, f = load(scores_dir, m, "nyc0"), load(scores_dir, m, "nycfix")
        if not z or not f:
            continue
        n01, n10, p = mcnemar(z, f)
        number(f"{m}_mcnemar_fix_p", round(p, 4), 4)
        rows_cmp.append({
            "Comparison": f"{labels[m]}: zero-shot vs 3-shot fixed",
            "Only B correct": n01, "Only A correct": n10,
            "p (exact McNemar)": f"{p:.3g}",
        })

    # The two leading models under retrieval, collapsed to one value per item.
    top = sorted(order, key=lambda m: _mean(load(scores_dir, m, "nycsim")), reverse=True)[:2]
    if len(top) == 2:
        a, b = (load(scores_dir, m, "nycsim") for m in top)
        ga, gb = defaultdict(list), defaultdict(list)
        for r in a:
            ga[r["id"]].append(r["strict"])
        for r in b:
            gb[r["id"]].append(r["strict"])
        common = sorted(set(ga) & set(gb))
        diff = np.array([np.mean(ga[i]) - np.mean(gb[i]) for i in common])
        tpl = {r["id"]: r["template_id"] for r in a}
        groups = defaultdict(list)
        for i, dv in zip(common, diff):
            groups[tpl[i]].append({"strict": dv, "id": i, "template_id": tpl[i]})
        lo, hi = cluster_boot(groups, _mean)
        number("top2_diff", round(100 * float(diff.mean()), 1), 1)
        number("top2_ci_lo", round(100 * lo, 1), 1)
        number("top2_ci_hi", round(100 * hi, 1), 1)
        number("top2_a", labels[top[0]])
        number("top2_b", labels[top[1]])
        rows_cmp.append({
            "Comparison": f"{labels[top[0]]} vs {labels[top[1]]} (3-shot retrieved)",
            "Only B correct": "-", "Only A correct": "-",
            "p (exact McNemar)": f"diff {100 * diff.mean():+.1f} pp, "
                                 f"95% CI [{100 * lo:.1f}, {100 * hi:.1f}]",
        })

    return rows_ci, rows_cmp


def make_contrast_table(scores_dir, order, labels, make_df):
    rows = pairwise_contrasts(scores_dir, order, labels)
    if not rows:
        return
    rows.sort(key=lambda r: -abs(r["diff"]))
    n_sep = sum(1 for r in rows if r["p_holm"] < 0.05)
    number("n_model_pairs", len(rows))
    number("n_model_pairs_separated", n_sep)
    out = []
    for r in rows:
        tag = f"{labels[r['a']]} - {labels[r['b']]}"
        key = f"{r['a']}__{r['b']}"
        number(f"contrast_{key}_diff", round(r["diff"], 1), 1)
        number(f"contrast_{key}_lo", round(r["lo"], 1), 1)
        number(f"contrast_{key}_hi", round(r["hi"], 1), 1)
        # A bootstrap p of 0 only means no replicate of N_BOOT crossed zero.
        # Store the resolution floor rather than a zero the prose would print.
        number(f"contrast_{key}_p_holm",
               float(f"{max(r['p_holm'], 1.0 / N_BOOT):.3g}"))
        # A bootstrap p of 0 means no replicate of N_BOOT crossed zero, which is
        # a resolution floor rather than evidence of an arbitrarily small p.
        # The string form also keeps the column from being re-rounded to one
        # decimal by the table formatter, which would show 0.055 as "0.1".
        p_str = "< 1e-4" if r["p_holm"] < 1e-4 else f"{r['p_holm']:.3g}"
        number(f"contrast_{key}_slo", round(100 * r["slo"], 1), 1)
        number(f"contrast_{key}_shi", round(100 * r["shi"], 1), 1)
        out.append({
            "Comparison": tag,
            "Diff (pp)": round(r["diff"], 1),
            "95% CI (per pair)": f"[{r['lo']:.1f}, {r['hi']:.1f}]",
            "95% CI (simultaneous)": f"[{100 * r['slo']:.1f}, {100 * r['shi']:.1f}]",
            "Holm p": p_str,
            "Separated": "yes" if r["p_holm"] < 0.05 else "no",
        })
    df = make_df(out, ["Comparison", "Diff (pp)", "95% CI (per pair)",
                       "95% CI (simultaneous)", "Holm p", "Separated"])
    save_table(df, "model_contrasts", floatfmt=".1f")
    print("\n== Table: model_contrasts ==")
    print(df.to_markdown(index=False, floatfmt=".1f"))


def make_tables(scores_dir, order, labels, make_df):
    make_contrast_table(scores_dir, order, labels, make_df)
    ci, cmp_ = run(scores_dir, order, labels)
    if ci:
        cols = ["Model", "S-sim %", "95% CI", "Gap (pp)", "Gap 95% CI", "Gap excludes 0"]
        df = make_df(ci, cols)
        save_table(df, "uncertainty", floatfmt=".1f")
        print("\n== Table: uncertainty ==")
        print(df.to_markdown(index=False, floatfmt=".1f"))
    if cmp_:
        cols = ["Comparison", "Only A correct", "Only B correct", "p (exact McNemar)"]
        df = make_df(cmp_, cols)
        save_table(df, "paired_tests", floatfmt=".3g")
        print("\n== Table: paired_tests ==")
        print(df.to_markdown(index=False, floatfmt=".3g"))
