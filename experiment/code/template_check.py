#!/usr/bin/env python3
"""Rendered-template check (PLAN 10 item 2). For each pilot prompt rendered on the PC with the
model's HF chat template, send it to Ollama as a raw /api/generate prompt first (fresh), then the
same messages through /api/chat. If Ollama's template renders the identical token sequence, the
greedy outputs are identical and the chat request reuses the whole prompt from the KV cache
(prompt_eval_count drops to ~1). Stdlib only.

    python3 template_check.py --model TAG --family qwen25coder15 --checks pilot_template_check.jsonl \
        --gen generation.json --out template_check_TAG.jsonl
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from run_ollama import chat

BOS_TEXT = {"prem1b": "<｜begin▁of▁sentence｜>", "llama32_3b": "<|begin_of_text|>"}  # tokenizer adds BOS itself


def generate_raw(host, model, prompt, gen):
    body = json.dumps({"model": model, "prompt": prompt, "raw": True, "stream": False,
                       "keep_alive": gen["keep_alive"], "options": gen["options"]}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--family", required=True)
    ap.add_argument("--checks", type=Path, required=True); ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--host", default="http://127.0.0.1:11434")
    a = ap.parse_args()
    gen = json.loads(a.gen.read_text())
    items = [c for c in map(json.loads, open(a.checks, encoding="utf-8")) if c["family"] == a.family]
    with open(a.out, "w", encoding="utf-8") as f:
        for c in items:
            p = c["hf_rendered"]
            bos = BOS_TEXT.get(a.family)
            if bos and p.startswith(bos):
                p = p[len(bos):]
            r1 = generate_raw(a.host, a.model, p, gen)
            r2 = chat(a.host, a.model, c["messages"], gen)
            rec = {"model": a.model, "rid": c["rid"], "cond": c["cond"], "hf_n_tokens": c["hf_n_tokens_incl_special"],
                   "raw_prompt_eval_count": r1.get("prompt_eval_count"), "chat_prompt_eval_count": r2.get("prompt_eval_count"),
                   "raw_out": r1.get("response"), "chat_out": r2["message"]["content"],
                   "same_output": r1.get("response") == r2["message"]["content"], "t_ns": time.time_ns()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(c["rid"], c["cond"], rec["same_output"], rec["hf_n_tokens"], rec["raw_prompt_eval_count"],
                  rec["chat_prompt_eval_count"], flush=True)


if __name__ == "__main__":
    main()
