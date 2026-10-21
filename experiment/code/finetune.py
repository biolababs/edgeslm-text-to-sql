#!/usr/bin/env python3
"""On-box LoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct on one fold's NYC pool (PLAN 7).

Revised from prior-code/finetune.py:
  - fold-aware data: nyc_pool items whose templates are NOT held out in fold F (pool F);
  - the training prompt is exactly the zero-shot chat prompt of PLAN 5.1 (prompts.messages_common),
    rendered with the tokenizer's chat template; loss on the SQL answer tokens only (answer_loss kept);
  - 1 epoch by default (D11); --max-steps for the pilot smoke test;
  - saves ONLY the adapter (no in-PyTorch merge, which ran out of memory in the exploratory pass);
  - per-step timing and peak memory logged to train_log.json.
Needs LD_LIBRARY_PATH=$HOME/bench/cudss/lib (the Jetson torch wheel links libcudss at import).

    python finetune.py --base ~/bench/hf/qwc15 --pool nyc_pool.jsonl --fold F1 --schema schema.sql --out train/f1
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from prompts import messages_common


def encode(tok, schema, ex, max_len):
    msgs = messages_common(schema, ex["question"])
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    answer_ids = tok(ex["sql"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return torch.tensor(ids), torch.tensor(labels)


def answer_loss(model, ids, labels):
    """Cross-entropy over answer tokens only, projecting just those hidden states to the vocab."""
    causal_lm = model.base_model.model
    hidden = causal_lm.model(input_ids=ids[None]).last_hidden_state[0]
    targets = labels[1:]
    keep = targets != -100
    logits = causal_lm.lm_head(hidden[:-1][keep]).float()
    return torch.nn.functional.cross_entropy(logits, targets[keep])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--fold", required=True, choices=["F1", "F2", "F3", "F4"])
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimiser steps (pilot)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    schema = args.schema.read_text(encoding="utf-8")
    with open(args.pool, encoding="utf-8") as f:
        pool = [json.loads(l) for l in f if l.strip()]
    train = sorted([p for p in pool if p["fold"] != args.fold], key=lambda p: p["id"])
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    data = [encode(tok, schema, ex, args.max_len) for ex in train]
    n_tok = sum(len(d[0]) for d in data)
    n_ans = sum(int((d[1] != -100).sum()) for d in data)
    print(f"fold {args.fold}: {len(data)} examples, mean length {n_tok / len(data):.0f} tokens", flush=True)

    steps_full = math.ceil(len(data) * args.epochs / args.grad_accum)
    steps = min(steps_full, args.max_steps) if args.max_steps else steps_full
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, round(steps_full * 0.05)), steps_full)

    model.train()
    log, step, running = [], 0, 0.0
    t0 = time.time(); t_step = t0; seen = 0; seen_ids = []
    for epoch in range(args.epochs):
        order = list(range(len(data)))
        random.shuffle(order)
        for i, j in enumerate(order, 1):
            ids, labels = data[j]
            loss = answer_loss(model, ids.cuda(), labels.cuda())
            (loss / args.grad_accum).backward()
            running += loss.item(); seen += 1; seen_ids.append(train[j]["id"])
            if i % args.grad_accum == 0 or i == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                k = args.grad_accum if i % args.grad_accum == 0 else i % args.grad_accum
                now = time.time()
                log.append({"step": step, "epoch": epoch, "loss": running / k, "elapsed_s": now - t0,
                            "step_s": now - t_step, "t_ns": time.time_ns(), "lr": sched.get_last_lr()[0]})
                t_step = now; running = 0.0
                if step % 5 == 0 or step == steps:
                    print(f"PROGRESS {step}/{steps} lora fold {args.fold} loss {log[-1]['loss']:.4f} {now - t0:.0f}s", flush=True)
                if step >= steps:
                    break
        if step >= steps:
            break

    train_s = time.time() - t0
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out / "adapter")
    (args.out / "train_log.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()}, "fold": args.fold, "n_examples": len(data),
        "examples_seen": seen, "seen_ids": seen_ids, "tokens_total": n_tok, "answer_tokens_total": n_ans,
        "steps": step, "steps_full_epoch": steps_full, "train_seconds": train_s,
        "s_per_step": train_s / max(step, 1), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "torch": torch.__version__, "log": log}, indent=1))
    print(f"done: {step} steps in {train_s:.0f} s, peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
