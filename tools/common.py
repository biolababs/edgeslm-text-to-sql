"""Shared helpers for analysis scripts. Everything the paper reports goes through
these functions so that every number, table and figure is regenerated from raw data."""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
BUILD = ROOT / "build"
FIGURES = BUILD / "figures"
TABLES = BUILD / "tables"

_numbers = {}


def number(key, value, digits=None):
    """Record a value the paper cites as {{key}}. Rounded once, here, so the
    paper and numbers.json always agree."""
    if digits is not None and isinstance(value, float):
        value = round(value, digits)
    _numbers[key] = value
    return value


def summarize(values, confidence=0.95):
    """Mean, sample std, n and a t-distribution confidence interval."""
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(xs)
    mean = sum(xs) / n if n else float("nan")
    if n < 2:
        return {"mean": mean, "std": float("nan"), "n": n, "ci_low": float("nan"), "ci_high": float("nan")}
    std = stats.tstd(xs)
    half = stats.t.ppf((1 + confidence) / 2, n - 1) * std / math.sqrt(n)
    return {"mean": mean, "std": std, "n": n, "ci_low": mean - half, "ci_high": mean + half}


def proportion_ci(correct, total, confidence=0.95):
    """Wilson score interval for accuracy-style results."""
    if total == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf((1 + confidence) / 2)
    p = correct / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return centre - half, centre + half


def save_figure(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_table(df, name, floatfmt=".2f"):
    """Writes CSV (for reuse) and a Markdown pipe table (for the paper)."""
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    md = df.to_markdown(index=False, floatfmt=floatfmt) if _has_tabulate() else _markdown(df, floatfmt)
    (TABLES / f"{name}.md").write_text(md + "\n", encoding="utf-8")


def write_numbers():
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "numbers.json").write_text(json.dumps(_numbers, indent=2, sort_keys=True), encoding="utf-8")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def _markdown(df, floatfmt):
    def fmt(v):
        return format(v, floatfmt) if isinstance(v, float) else str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])
