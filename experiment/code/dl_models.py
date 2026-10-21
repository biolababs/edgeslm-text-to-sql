#!/usr/bin/env python3
"""Download the pinned Hugging Face revisions used for our own GGUF conversions (PLAN 2.1, 2.2),
then verify every LFS file's SHA-256 against the Hub API. Run in the PC convert env.

    python dl_models.py --out C:/Users/biola/paper/models-cache/c9/hf
"""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODELS = {
    "xiyan3b": ("XGenerationLab/XiYanSQL-QwenCoder-3B-2504", "b883e58ed83f74bab037d6a7b90c4b8706d357d7"),
    "prem1b": ("prem-research/prem-1B-SQL", "44dd7fcf9227af4efed936bf29323c61bf66aad1"),
    "qc15": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "2e1fd397ee46e1388853d2af2c993145b0f1098a"),
}
# Prem ships optimizer/scheduler/trainer state (10.7 GB) that conversion does not need.
IGNORE = ["optimizer.pt", "scheduler.pt", "trainer_state.json", "*.bin", "*.pth", "*.onnx", "*.gguf"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    api = HfApi()
    report = {}
    for key, (repo, rev) in MODELS.items():
        if args.only and key not in args.only:
            continue
        d = args.out / key
        snapshot_download(repo, revision=rev, local_dir=d, ignore_patterns=IGNORE)
        info = api.model_info(repo, revision=rev, files_metadata=True)
        files = {}
        for s in info.siblings:
            p = d / s.rfilename
            if not p.exists():
                continue
            got = sha256(p)
            exp = s.lfs.sha256 if s.lfs else None
            files[s.rfilename] = {"size": p.stat().st_size, "sha256": got, "hub_lfs_sha256": exp,
                                  "ok": (exp is None) or (exp == got)}
        report[key] = {"repo": repo, "revision": rev, "resolved_sha": info.sha,
                       "license": (info.card_data or {}).get("license") if info.card_data else None,
                       "files": files, "all_ok": all(f["ok"] for f in files.values())}
        print(key, "all_ok", report[key]["all_ok"], flush=True)
    (args.out / "download_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
