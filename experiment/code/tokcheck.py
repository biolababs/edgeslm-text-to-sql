#!/usr/bin/env python3
"""Prompt-length gate and rendered-prompt examples (PLAN 5.5 context rule, PLAN 10 items 2 and 4).

For every request file and every model family, render the messages with that model's HF chat
template (add_generation_prompt=True), count tokens, and flag any prompt with
n_tokens + num_predict > num_ctx. Also writes one rendered example per condition and model
to experiment/prompts/rendered_examples.txt, and the HF-rendered pilot prompts used by the
on-board raw-vs-chat template check (pilot_template_check.jsonl).

    python tokcheck.py --requests experiment/requests --tok-dir models-cache/c9/tok --out experiment/prompts
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

FAMILIES = {  # model tags -> tokenizer source (HF repo id or local dir)
    "qwen25coder15": ["qc15-q4-ollama", "qc15-f16-ours", "qc15-q8-ours", "qc15-q4-ours"],
    "qwen25coder3": ["qc3-q4-ollama"],
    "llama32_3b": ["llama32-3b-q4-ollama"],
    "qwen3_4b_2507": ["qwen3-4b-q4-ollama"],
    "xiyan3b": ["xiyan3b-q4-ours"],
    "prem1b": ["prem1b-q4-ours"],
}
NUM_CTX, NUM_PREDICT = 4096, 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--tok-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    files = sorted(p for p in args.requests.glob("*.jsonl") if not p.name.startswith("pilot_"))
    report, examples, tmpl_check = {}, [], []
    pilot = {p.stem: [json.loads(l) for l in open(p, encoding="utf-8")] for p in args.requests.glob("pilot_*.jsonl")}
    for fam, tags in FAMILIES.items():
        tok = AutoTokenizer.from_pretrained(str(args.tok_dir / fam))
        fr = {}
        for p in files:
            if fam != "xiyan3b" and p.stem in ("nycnat", "curnat", "birdnat"):
                continue
            recs = [json.loads(l) for l in open(p, encoding="utf-8")]
            counts, over = [], []
            for i, r in enumerate(recs):
                text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
                n = len(tok(text, add_special_tokens=False)["input_ids"])
                counts.append(n)
                if n + NUM_PREDICT > NUM_CTX:
                    over.append({"rid": r["rid"], "n_tokens": n})
                if i == 0:
                    examples.append(f"===== model family {fam} ({', '.join(tags)}) | condition {p.stem} | rid {r['rid']} | {n} tokens\n{text}\n")
            fr[p.stem] = {"n": len(counts), "max_tokens": max(counts), "mean_tokens": sum(counts) / len(counts), "over_limit": over}
        report[fam] = {"tags": tags, "files": fr}
        for name, recs in pilot.items():
            if fam != "xiyan3b" and name.endswith("nat"):
                continue
            for r in recs[:2]:
                text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
                tmpl_check.append({"family": fam, "tags": tags, "rid": r["rid"], "cond": r["cond"], "messages": r["messages"],
                                   "hf_rendered": text, "hf_n_tokens_incl_special": len(tok(text, add_special_tokens=False)["input_ids"])})
        print(fam, {k: (v["max_tokens"], len(v["over_limit"])) for k, v in fr.items()}, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "token_lengths.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (args.out / "rendered_examples.txt").write_text("\n".join(examples), encoding="utf-8")
    with open(args.requests / "pilot_template_check.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for t in tmpl_check:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    tot = sum(len(v["over_limit"]) for r in report.values() for v in r["files"].values())
    print("prompts over num_ctx - num_predict:", tot)


if __name__ == "__main__":
    main()
