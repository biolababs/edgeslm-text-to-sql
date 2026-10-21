"""Catches the one class of error that placeholders cannot catch.

Every number in the paper comes from build/numbers.json, so no number can be
wrong. What has repeatedly been wrong is the *prose around* the numbers:
superlatives, ordinals and universal quantifiers that were typed by hand and
never checked ("the weakest model", "fourth on Spider", "every model gains at
least", "the narrowest interval").

This tool finds every such sentence, prints the true ranking beside it, and
refuses to pass until each one has been explicitly reviewed and recorded in
paper/claims.reviewed.json. A reviewed sentence is pinned by hash, so editing
it forces a fresh review.

    python tools/audit_claims.py              # audit, non-zero exit if unreviewed
    python tools/audit_claims.py --rankings   # print the ranking tables
    python tools/audit_claims.py --accept "<reason>"   # record all current claims
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "build" / "paper.md"
NUMBERS = ROOT / "build" / "numbers.json"
REVIEWED = ROOT / "paper" / "claims.reviewed.json"

CONFIG = ROOT / "paper" / "audit.config.json"


def load_config():
    """Study-specific names live in a config file, not in this script, so the
    tool can be pointed at another paper without editing it."""
    if not CONFIG.exists():
        sys.exit(f"missing {CONFIG.relative_to(ROOT)} - it names the models, the "
                 f"metrics to rank them on, and the terms that make an ordinal "
                 f"or superlative a ranking claim")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    models = cfg["models"]
    metrics = {k: (v["label"], v["higher_is_better"]) for k, v in cfg["metrics"].items()}
    subject = re.compile(r"\b(" + "|".join(cfg["subject_terms"]) + r")\b", re.I)
    return models, metrics, subject

TRIGGERS = [
    (r"\b(highest|lowest|best|worst|weakest|strongest|largest|smallest|"
     r"narrowest|widest|fastest|slowest|cheapest|costliest|leads|tops|"
     r"outperforms|the most|the least)\b", "superlative"),
    (r"\b(first|second|third|fourth|fifth|sixth)\b(?!\s+(step|part|point|"
     r"category|paragraph|section|author|person|language))", "ordinal"),
    # A universal is checkable whatever noun it quantifies, so match the
    # quantifier rather than a list of nouns. Matching only "every model" let
    # "every template-clustered interval" and "All taxi intervals" through.
    (r"\b(?:every|each)\s+\w+", "universal"),
    (r"\ball\s+(?:six|of\s+the|the|\d+|[a-z]+)\b", "universal"),
    (r"\bno(?:ne)?\s+(?!longer\b|doubt\b|one\b)\w+", "universal"),
    (r"\b(always|never|without exception|in every case|universally|"
     r"uniformly|throughout)\b", "universal"),
    (r"\b(only|solely|exclusively|entirely|purely)\b", "exclusivity"),
    (r"\bat (?:least|most)\b", "bound"),
]


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def sentences(text):
    body, out = [], []
    for line in text.splitlines():
        if line.startswith(("|", "![", "#", "---", ":")) or not line.strip():
            continue
        body.append(line)
    for para in " ".join(body).split(". "):
        p = norm(para)
        if len(p) > 25:
            out.append(p if p.endswith(".") else p + ".")
    return out


def rank(nums, key, higher, models):
    vals = [(m, nums.get(f"{m}_{key}")) for m in models]
    vals = [(m, v) for m, v in vals if isinstance(v, (int, float))]
    return sorted(vals, key=lambda t: -t[1] if higher else t[1])


def print_rankings(nums, models, metrics):
    for key, (desc, higher) in metrics.items():
        r = rank(nums, key, higher, models)
        if not r:
            continue
        order = "high to low" if higher else "low to high"
        print(f"\n{desc}  [{key}]  ({order})")
        for i, (m, v) in enumerate(r, 1):
            print(f"   {i}. {models[m]:16s} {v}")


# An ordinal or superlative is only a ranking claim if it is about something
# rankable. A universal quantifier is risky wherever it appears.
ALWAYS = {"universal"}


def classify(s, subject):
    kinds = sorted({k for pat, k in TRIGGERS if re.search(pat, s, re.I)})
    if not kinds:
        return []
    if set(kinds) & ALWAYS:
        return kinds
    return kinds if subject.search(s) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", action="store_true")
    ap.add_argument("--accept", metavar="REASON")
    a = ap.parse_args()

    if not PAPER.exists():
        sys.exit("build/paper.md not found - run tools/render.py first")
    nums = json.loads(NUMBERS.read_text(encoding="utf-8"))
    models, metrics, subject = load_config()

    if a.rankings:
        print_rankings(nums, models, metrics)
        return

    reviewed = json.loads(REVIEWED.read_text(encoding="utf-8")) if REVIEWED.exists() else {}
    flagged = []
    for s in sentences(PAPER.read_text(encoding="utf-8")):
        kinds = classify(s, subject)
        if kinds:
            flagged.append((hashlib.sha256(s.encode()).hexdigest()[:16], kinds, s))

    unreviewed = [f for f in flagged if f[0] not in reviewed]

    if a.accept:
        for h, kinds, s in flagged:
            reviewed.setdefault(h, {"kinds": kinds, "reason": a.accept, "text": s[:300]})
        REVIEWED.parent.mkdir(parents=True, exist_ok=True)
        REVIEWED.write_text(json.dumps(reviewed, indent=1, sort_keys=True), encoding="utf-8")
        print(f"Recorded {len(flagged)} claims as reviewed ({len(unreviewed)} newly).")
        return

    print(f"{len(flagged)} ranking/quantifier claims found, "
          f"{len(flagged) - len(unreviewed)} previously reviewed.\n")
    if not unreviewed:
        print("CLAIMS_AUDIT_OK")
        return
    for h, kinds, s in unreviewed:
        print(f"[{h}] ({', '.join(kinds)})\n    {s}\n")
    print(f"{len(unreviewed)} claim(s) need checking against the data.")
    print("Run with --rankings to see the true ordering, then --accept "
          '"<who checked, when>" once each is verified.')
    sys.exit(1)


if __name__ == "__main__":
    main()
