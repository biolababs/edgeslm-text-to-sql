#!/usr/bin/env python3
"""Send pre-rendered chat requests to Ollama and record outputs and per-query timing (PLAN 5, 6.2).

Revised from prior-code/run_ollama.py:
  - prompts are NOT built here: each request record already holds its "messages"
    (experiment/code/build_requests.py on the PC), so every model sees identical text;
  - every sampling option is sent explicitly (config/generation.json), including repeat_penalty 1.0;
  - client-side time.time_ns() at send and receive, plus all Ollama timing fields;
  - the server-reported model must equal the requested tag (assert);
  - SQL is extracted with the fixed rule of PLAN 5.5 (prompts.extract_sql) and the rule is recorded;
  - cap hits (done_reason == "length") and prompt_eval_count + num_predict > num_ctx are flagged.
Resumes by request id (rid). Stdlib only (Jetson system Python 3.10).

    python3 run_ollama.py --model qc15-q4-ollama --requests req/pilot_nyc0.jsonl --out acc/x.jsonl \
        --gen config/generation.json [--warmup req/warmup.jsonl] [--tag-note ...]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from prompts import extract_sql


def chat(host, model, messages, gen):
    body = json.dumps({"model": model, "messages": messages, "stream": False, "think": gen["think"],
                       "keep_alive": gen["keep_alive"], "options": gen["options"]}).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Ollama tag")
    ap.add_argument("--requests", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--warmup", type=Path, help="requests JSONL; first N sent untimed and discarded")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    gen = json.loads(args.gen.read_text(encoding="utf-8"))
    num_ctx, num_predict = gen["options"]["num_ctx"], gen["options"]["num_predict"]
    reqs = []
    for p in args.requests:
        with open(p, encoding="utf-8") as f:
            reqs += [json.loads(l) for l in f if l.strip()]
    if args.limit:
        reqs = reqs[: args.limit]
    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(l)["rid"] for l in f if l.strip()}
    todo = [r for r in reqs if r["rid"] not in done]
    print(f"{args.model}: {len(todo)} to do, {len(done)} already done", flush=True)

    if args.warmup and todo:
        with open(args.warmup, encoding="utf-8") as f:
            wu = [json.loads(l) for l in f if l.strip()][: gen.get("warmup_requests", 5)]
        for w in wu:
            r = chat(args.host, args.model, w["messages"], gen)
            assert r.get("model") == args.model, (r.get("model"), args.model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            t_send = time.time_ns()
            r = chat(args.host, args.model, q["messages"], gen)
            t_recv = time.time_ns()
            if r.get("model") != args.model:
                sys.exit(f"server answered with model {r.get('model')!r}, requested {args.model!r}")
            raw = r["message"]["content"]
            sql, rule = extract_sql(raw)
            pe = r.get("prompt_eval_count") or 0
            rec = {"rid": q["rid"], "id": q["id"], "cond": q["cond"], "set": q.get("set"), "model": r.get("model"),
                   "raw": raw, "sql": sql, "extract_rule": rule,
                   "t_send_ns": t_send, "t_recv_ns": t_recv, "wall_s": (t_recv - t_send) / 1e9,
                   "prompt_eval_count": r.get("prompt_eval_count"), "prompt_eval_duration": r.get("prompt_eval_duration"),
                   "eval_count": r.get("eval_count"), "eval_duration": r.get("eval_duration"),
                   "load_duration": r.get("load_duration"), "total_duration": r.get("total_duration"),
                   "done_reason": r.get("done_reason"),
                   "truncated": r.get("done_reason") == "length",
                   "ctx_flag": pe + num_predict > num_ctx}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i == 1 or i % 25 == 0 or i == len(todo):
                print(f"PROGRESS {i}/{len(todo)} {args.model} {q['cond']} {rec['wall_s']:.2f}s {rec['eval_count']} tok", flush=True)


if __name__ == "__main__":
    main()
