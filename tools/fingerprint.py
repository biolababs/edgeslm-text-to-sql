"""Records SHA-256 hashes of every raw result file and the software environment.
  python tools/fingerprint.py          write results/MANIFEST.sha256 + build/environment.json
  python tools/fingerprint.py --check  fail if any raw file was changed, added or removed"""
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
MANIFEST = ROOT / "results" / "MANIFEST.sha256"


def hashes():
    out = {}
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and p.name != ".gitkeep":
            out[p.relative_to(RAW).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def read_manifest():
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, path = line.split("  ", 1)
        entries[path] = digest
    return entries


def environment():
    reqs = (ROOT / "requirements.txt").read_text().split()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {r: _version(r) for r in reqs},
    }


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def main():
    current = hashes()
    if "--check" in sys.argv:
        if not MANIFEST.exists():
            sys.exit("No results/MANIFEST.sha256 yet. Run: python tools/fingerprint.py")
        expected = read_manifest()
        problems = [f"changed: {p}" for p in expected if p in current and current[p] != expected[p]]
        problems += [f"missing: {p}" for p in expected if p not in current]
        problems += [f"not in manifest: {p}" for p in current if p not in expected]
        if problems:
            print("Raw data does NOT match the manifest:\n  " + "\n  ".join(problems))
            sys.exit(1)
        print(f"Raw data verified: {len(current)} files match results/MANIFEST.sha256")
        return
    if not current:
        sys.exit("results/raw is empty. Put your benchmark results there first.")
    MANIFEST.write_text("".join(f"{d}  {p}\n" for p, d in current.items()), encoding="utf-8")
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "environment.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    print(f"Fingerprinted {len(current)} raw files -> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
