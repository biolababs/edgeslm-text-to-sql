"""One command to rebuild and confirm the paper from the raw results.
  python reproduce.py            verify data, rerun analysis, confirm numbers, check citations, build paper
  python reproduce.py --lock     accept the current numbers as the published ones (authors only)
  python reproduce.py --offline  skip the online citation check"""
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUMBERS = ROOT / "build" / "numbers.json"
LOCK = ROOT / "paper" / "numbers.lock.json"
REL_TOL = 1e-9


def run(step, *cmd):
    print(f"\n== {step}", flush=True)
    return subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode == 0


def same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))
    return a == b


def confirm_numbers(lock):
    print("\n== Confirm numbers against the published values")
    current = json.loads(NUMBERS.read_text(encoding="utf-8"))
    if lock:
        LOCK.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Locked {len(current)} values -> paper/numbers.lock.json")
        return True
    if not LOCK.exists():
        print("No paper/numbers.lock.json yet. Authors: run `python reproduce.py --lock` once the paper is final.")
        return False
    published = json.loads(LOCK.read_text(encoding="utf-8"))
    diffs = [f"{k}: published {published.get(k)!r}, reproduced {current.get(k)!r}"
             for k in sorted(published.keys() | current.keys()) if not same(published.get(k), current.get(k))]
    if diffs:
        print("Numbers differ from the published paper:\n  " + "\n  ".join(diffs))
        return False
    print(f"All {len(published)} reported values reproduced exactly")
    return True


def main():
    results = {
        "Raw data unchanged": run("Verify raw data", "tools/fingerprint.py", "--check"),
        "Analysis ran": run("Rerun analysis", "analysis/analyze.py"),
    }
    results["Numbers match paper"] = results["Analysis ran"] and confirm_numbers("--lock" in sys.argv)
    if "--offline" not in sys.argv:
        results["Citations verified"] = run("Verify citations", "tools/verify_citations.py")
    results["Paper built"] = results["Analysis ran"] and run("Build paper", "tools/render.py")

    print("\n== Summary")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
