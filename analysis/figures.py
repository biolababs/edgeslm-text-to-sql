"""Figures for the c9 text-to-SQL paper. Called from analyze.py after the tables
are built, so the summary dict already carries the derived seen/unseen splits.

Every figure is drawn from the same summary.json the tables use, so a figure can
never disagree with a table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

from scipy import stats  # noqa: E402

from common import number, save_figure  # noqa: E402

# Okabe-Ito colourblind-safe palette, one colour per model.
COLOURS = {
    "prem1b-q4-ours": "#0072B2",
    "qc15-q4-ollama": "#E69F00",
    "qc3-q4-ollama": "#D55E00",
    "xiyan3b-q4-ours": "#56B4E9",
    "llama32-3b-q4-ollama": "#009E73",
    "qwen3-4b-q4-ollama": "#CC79A7",
}
MARKERS = {"SQL": "o", "Code": "s", "General": "^"}
GREY = "#4D4D4D"
LIGHT = "#E8E8E8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": GREY,
    "axes.linewidth": 0.8,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "legend.frameon": False,
    "figure.facecolor": "white",
})


def _pct(x):
    return None if x is None else 100 * x


def _spine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _params(s):
    return float(s.rstrip("B"))


# ──────────────────────────────────────────────────────────────────────
# Figure 1. Experimental design map (relationship map)
# ──────────────────────────────────────────────────────────────────────
def design_map(labels, model_type):
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge, size=8, weight="normal"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
            facecolor=face, edgecolor=edge, linewidth=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, weight=weight, color="#111111", linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
            color="#9A9A9A", linewidth=0.7, shrinkA=0, shrinkB=0))

    headers = [(1.1, "Question sets"), (3.5, "Prompting"), (6.0, "Models (Q4)"), (8.6, "Measures")]
    for x, t in headers:
        ax.text(x, 5.62, t, ha="center", va="center", fontsize=9.5,
                weight="bold", color="#111111")

    # Column 1: question sets
    sets = [
        ("NYC taxi\n449 eval / 744 pool\n26 templates, 4 folds", 4.05),
        ("Curated\n92 questions\nAI-drafted, checked", 2.75),
        ("Spider dev 1,034\nBIRD Mini-Dev 500", 1.45),
    ]
    for t, y in sets:
        box(0.15, y, 1.9, 1.0, t, "#F4F8FB", "#9DBFD6")

    # Column 2: prompting conditions
    conds = [
        ("Zero-shot", 4.25),
        ("3-shot retrieved\n(S-similar)", 3.05),
        ("3-shot fixed\n(S-fixed)", 1.85),
    ]
    for t, y in conds:
        box(2.65, y, 1.7, 0.85, t, "#FFF8EE", "#E0B877")

    # Column 3: models, grouped in one panel so arrows meet an edge, not a point
    ax.add_patch(FancyBboxPatch(
        (4.88, 1.36), 2.24, 3.80, boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="#FCFCFC", edgecolor="#D8D8D8", linewidth=0.8, zorder=0))
    for i, (mid, lab) in enumerate(labels.items()):
        y = 4.55 - i * 0.62
        box(4.95, y, 2.1, 0.48, f"{lab}  ·  {model_type[mid]}",
            "#FFFFFF", COLOURS[mid], size=7.6)

    # Column 4: measures
    meas = [
        ("Correctness\nstrict execution match", 4.15),
        ("Latency\nwall time, tok/s", 2.85),
        ("Energy\nJ/query, J/correct", 1.55),
    ]
    for t, y in meas:
        box(7.75, y, 1.95, 0.95, t, "#F2FAF6", "#84C5AC")

    # NYC and curated sets run under all three prompting conditions; the public
    # benchmarks are zero-shot only.
    for y1 in (4.55, 3.25):
        for y2 in (4.675, 3.475, 2.275):
            arrow(2.05, y1, 2.65, y2)
    arrow(2.05, 1.95, 2.65, 4.675)
    ax.text(1.1, 1.28, "zero-shot only", ha="center", va="center",
            fontsize=7, style="italic", color=GREY)
    for y in (4.675, 3.475, 2.275):
        arrow(4.35, y, 4.88, y)
    for y in (4.625, 3.325, 2.025):
        arrow(7.12, y, 7.75, y)

    ax.add_patch(FancyBboxPatch(
        (0.15, 0.18), 9.55, 0.48, boxstyle="round,pad=0.04,rounding_size=0.06",
        facecolor="#FAFAFA", edgecolor="#CFCFCF", linewidth=0.8))
    ax.text(5.0, 0.42, "NVIDIA Jetson Orin Nano Super · 8 GB · MAXN_SUPER · "
                       "Ollama 0.30.11 · one request at a time",
            ha="center", va="center", fontsize=8, style="italic", color=GREY)

    save_figure(fig, "design_map")


# ──────────────────────────────────────────────────────────────────────
# Figure 2. Prompting effect on the taxi database
# ──────────────────────────────────────────────────────────────────────
def prompting_effect(S, order, labels):
    # The pooled S-similar figure averages the two exposures in equal numbers,
    # so showing it as one bar hides the paper's main result. The retrieved
    # condition is split into its withheld and covered halves instead.
    def get(cond, field="strict"):
        return lambda m: _pct(S.get(f"{m}/{cond}", {}).get(field))

    conds = [("Zero-shot", "#C9D6DF", get("nyc0")),
             ("3-shot fixed", "#9CBBD0", get("nycfix")),
             ("3-shot retrieved,\ntemplate withheld", "#4C8FB8", get("nycsim", "strict_unseen")),
             ("3-shot retrieved,\ntemplate present", "#00527A", get("nycsim", "strict_seen"))]
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    w = 0.2
    xs = range(len(order))
    for j, (lab, col, fn) in enumerate(conds):
        vals = [fn(m) for m in order]
        pos = [x + (j - 1.5) * w for x in xs]
        bars = ax.bar(pos, vals, w, label=lab, color=col,
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            if v is not None:
                ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=6.8, color=GREY)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([labels[m] for m in order], rotation=18, ha="right")
    ax.set_ylabel("Strict execution accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Examples help every model; how much depends on template coverage",
                 loc="left", weight="bold")
    ax.yaxis.grid(True, color=LIGHT, linewidth=0.7)
    ax.set_axisbelow(True)
    _spine(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=4, fontsize=7.6)
    save_figure(fig, "prompting_effect")


# ──────────────────────────────────────────────────────────────────────
# Figure 3. Template generalisation: seen vs unseen
# ──────────────────────────────────────────────────────────────────────
def generalisation_gap(S, order, labels):
    rows = []
    for m in order:
        d = S.get(f"{m}/nycsim", {})
        s, u = d.get("strict_seen"), d.get("strict_unseen")
        if s is not None and u is not None:
            # Round each share before differencing so the gap matches the table.
            rows.append((m, round(_pct(s), 1), round(_pct(u), 1)))
    if not rows:
        return
    rows.sort(key=lambda r: r[1] - r[2])

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    for i, (m, s, u) in enumerate(rows):
        ax.plot([u, s], [i, i], color="#CCCCCC", linewidth=2.2,
                solid_capstyle="round", zorder=1)
        ax.scatter(u, i, s=58, color="white", edgecolor=COLOURS[m],
                   linewidth=1.8, zorder=3)
        ax.scatter(s, i, s=58, color=COLOURS[m], edgecolor="white",
                   linewidth=0.8, zorder=3)
        ax.text(max(s, u) + 1.6, i, f"{s - u:+.1f} pp", va="center",
                fontsize=7.6, color=GREY)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([labels[m] for m, _, _ in rows])
    ax.set_xlabel("Strict execution accuracy under 3-shot retrieval (%)")
    gaps = [s - u for _, s, u in rows]
    lo_g, hi_g = min(gaps), max(gaps)
    lo_x = max(0, min(u for _, _, u in rows) - 6)
    ax.set_xlim(lo_x, 102)
    ax.set_title(f"Withholding the matching template costs {lo_g:.0f} to {hi_g:.0f} points, "
                 f"for every model", loc="left", weight="bold")
    ax.xaxis.grid(True, color=LIGHT, linewidth=0.7)
    ax.set_axisbelow(True)
    _spine(ax)
    ax.scatter([], [], s=58, color=GREY, edgecolor="white",
               label="Own template present in pool")
    ax.scatter([], [], s=58, color="white", edgecolor=GREY, linewidth=1.8,
               label="Own template withheld")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=8)
    save_figure(fig, "generalisation_gap")


# ──────────────────────────────────────────────────────────────────────
# Figure 4. Accuracy against energy per correct answer
# ──────────────────────────────────────────────────────────────────────
def accuracy_energy(S, order, labels, model_type):
    pts = []
    for m in order:
        d = S.get(f"{m}/nycsim", {})
        ex, ej = d.get("strict"), d.get("energy_j_mean")
        if ex and ej:
            pts.append((m, ej / ex, _pct(ex)))
    if not pts:
        return

    # Nudge labels that would otherwise collide.
    nudge = {"qc3-q4-ollama": (-30, 6), "xiyan3b-q4-ours": (26, 6)}

    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    for m, jpc, acc in pts:
        ax.scatter(jpc, acc, s=130, color=COLOURS[m],
                   marker=MARKERS[model_type[m]], edgecolor="white",
                   linewidth=1.1, zorder=3)
        ax.annotate(labels[m], (jpc, acc), textcoords="offset points",
                    xytext=nudge.get(m, (0, 12)), ha="center",
                    fontsize=7.8, color="#111111")

    front = sorted(pts, key=lambda p: p[1])
    best, keep = -1, []
    for p in front:
        if p[2] > best:
            keep.append(p)
            best = p[2]
    if len(keep) > 1:
        ax.plot([p[1] for p in keep], [p[2] for p in keep], "--",
                color="#AAAAAA", linewidth=1.0, zorder=1)
        mid = keep[len(keep) // 2]
        ax.annotate("efficiency frontier", (mid[1], mid[2]),
                    textcoords="offset points", xytext=(18, -26),
                    fontsize=7.6, style="italic", color=GREY,
                    arrowprops=dict(arrowstyle="-", color="#BBBBBB",
                                    linewidth=0.7))

    ax.set_xlabel("Energy per correct answer (J, lower is better)")
    ax.set_ylabel("Strict execution accuracy (%)")
    ax.set_title("Accuracy and energy cost, 3-shot retrieval on the taxi database",
                 loc="left", weight="bold")
    ax.grid(True, color=LIGHT, linewidth=0.7)
    ax.set_axisbelow(True)
    _spine(ax)
    for t, mk in MARKERS.items():
        ax.scatter([], [], marker=mk, s=70, color=GREY, label=t)
    ax.legend(title="Model type", loc="lower right", fontsize=8, title_fontsize=8)
    save_figure(fig, "accuracy_energy")


# ──────────────────────────────────────────────────────────────────────
# Figure 5. Benchmark accuracy against local-database accuracy
# ──────────────────────────────────────────────────────────────────────
def benchmark_vs_local(S, order, labels, model_type):
    pts = []
    for m in order:
        sp = S.get(f"{m}/spider0", {}).get("spider", {}).get("ex")
        nyc = S.get(f"{m}/nyc0", {}).get("strict")
        if sp is not None and nyc is not None:
            pts.append((m, _pct(sp), _pct(nyc)))
    if not pts:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    for m, sp, nyc in pts:
        ax.scatter(sp, nyc, s=130, color=COLOURS[m],
                   marker=MARKERS[model_type[m]], edgecolor="white",
                   linewidth=1.1, zorder=3)
        ax.annotate(labels[m], (sp, nyc), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=7.8, color="#111111")
    rho, p = stats.spearmanr([p[1] for p in pts], [p[2] for p in pts])
    number("spider_nyc_spearman", round(float(rho), 2), 2)
    number("spider_nyc_spearman_p", round(float(p), 2), 2)
    ax.text(0.03, 0.95, f"Spearman $\\rho$ = {rho:.2f}  (p = {p:.2f}, n = {len(pts)})",
            transform=ax.transAxes, fontsize=8, color=GREY, va="top")

    ax.set_xlabel("Spider dev execution accuracy (%)")
    ax.set_ylabel("Taxi database, zero-shot (%)")
    ax.set_xlim(55, 92)
    ax.set_ylim(0, 40)
    ax.set_title("Benchmark standing does not order the models on a new database",
                 loc="left", weight="bold")
    ax.grid(True, color=LIGHT, linewidth=0.7)
    ax.set_axisbelow(True)
    _spine(ax)
    for t, mk in MARKERS.items():
        ax.scatter([], [], marker=mk, s=70, color=GREY, label=t)
    ax.legend(title="Model type", loc="upper right", fontsize=8, title_fontsize=8)
    save_figure(fig, "benchmark_vs_local")


# ──────────────────────────────────────────────────────────────────────
# Figure 6. Error composition, zero-shot
# ──────────────────────────────────────────────────────────────────────
def error_composition(S, order, labels):
    cats = [("correct", "Correct", "#009E73"),
            ("wrong_result", "Wrong result", "#E69F00"),
            ("exec_error", "Execution error", "#D55E00"),
            ("truncated", "Truncated", "#8C8C8C"),
            ("no_sql", "No SQL", "#444444")]
    rows = [(m, S[f"{m}/nyc0"]["classes"], S[f"{m}/nyc0"]["n"])
            for m in order if S.get(f"{m}/nyc0", {}).get("classes")]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ys = range(len(rows))
    left = [0.0] * len(rows)
    for key, lab, col in cats:
        vals = [100 * c.get(key, 0) / n for _, c, n in rows]
        ax.barh(list(ys), vals, left=left, height=0.62, label=lab,
                color=col, edgecolor="white", linewidth=0.6)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 6:
                ax.text(l + v / 2, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white", weight="bold")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(list(ys))
    ax.set_yticklabels([labels[m] for m, _, _ in rows])
    ax.set_xlabel("Share of the 449 zero-shot taxi predictions (%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_title("Where zero-shot predictions fail", loc="left", weight="bold")
    _spine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, fontsize=8)
    save_figure(fig, "error_composition")


# ──────────────────────────────────────────────────────────────────────
# Figure 7. Speed against model size
# ──────────────────────────────────────────────────────────────────────
def latency_size(S, order, labels, params):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.4))
    for m in order:
        d = S.get(f"{m}/nyc0", {})
        p = _params(params[m])
        if d.get("decode_tok_s_mean"):
            a1.scatter(p, d["decode_tok_s_mean"], s=105, color=COLOURS[m],
                       edgecolor="white", linewidth=1.0, zorder=3)
        if d.get("wall_s_mean"):
            a2.scatter(p, d["wall_s_mean"], s=105, color=COLOURS[m],
                       edgecolor="white", linewidth=1.0, zorder=3)
    for ax, ylab, title in ((a1, "Decode throughput (tok/s)", "Generation speed"),
                            (a2, "Mean wall time per query (s)", "Response time")):
        ax.set_xlabel("Parameters (B)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", weight="bold", fontsize=9.5)
        ax.grid(True, color=LIGHT, linewidth=0.7)
        ax.set_axisbelow(True)
        _spine(ax)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                          markerfacecolor=COLOURS[m], markeredgecolor="white",
                          label=labels[m]) for m in order]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.8,
               bbox_to_anchor=(0.5, -0.17))
    save_figure(fig, "latency_size")


def make_all(S, order, labels, model_type, params):
    design_map(labels, model_type)
    prompting_effect(S, order, labels)
    generalisation_gap(S, order, labels)
    accuracy_energy(S, order, labels, model_type)
    benchmark_vs_local(S, order, labels, model_type)
    error_composition(S, order, labels)
    latency_size(S, order, labels, params)
    print("FIGURES_DONE 7 figures -> build/figures/")
