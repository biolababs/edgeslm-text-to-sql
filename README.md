<p align="center">
  <h1 align="center">Edge SLM Text-to-SQL</h1>
  <p align="center">
    <b>Accuracy, Generalisation, and Energy Cost of Small Open-Weight Models<br>on the NVIDIA Jetson Orin Nano Super</b>
  </p>
  <p align="center">
    <a href="#reproduction">Reproduce</a> &nbsp;·&nbsp;
    <a href="paper/paper.md">Paper</a> &nbsp;·&nbsp;
    <a href="#benchmarks">Benchmarks</a> &nbsp;·&nbsp;
    <a href="#citation">Cite</a>
  </p>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.12+-yellow.svg" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/Platform-Jetson_Orin_Nano_Super-green.svg" alt="Platform: Jetson Orin Nano Super">
  <img src="https://img.shields.io/badge/Models-1B–4B_Q4-orange.svg" alt="Models: 1B–4B Q4">
</p>

---

## Overview

Can a small language model translate natural-language questions into correct SQL on an 8 GB edge board — and what does each correct answer actually cost?

This repository contains data, code, and instructions to **reproduce every number** in the accompanying paper. Six open-weight models (1 B – 4 B parameters, 4-bit quantised) are evaluated across three workloads — a single-domain taxi database with held-out query templates, the Spider development set, and BIRD Mini-Dev v1 — with every accuracy figure paired to per-query latency and energy drawn from the Jetson's on-board power sensor.

### Key findings

> **Results will appear here once scoring and verification are complete.**
> All numbers in the paper are generated from raw data in this repository;
> `reproduce.py` regenerates and checks every one.

## Research Questions

| # | Question | Evaluation |
|:-:|---------|-----------|
| **RQ1** | How accurately do the models answer taxi queries, and how does accuracy change when query templates are withheld from the example pool? | Strict EX on 449 items × 4 held-out folds |
| **RQ2** | How do the same artifacts score on published benchmarks? | Spider EX / TS / EM; BIRD EX / Soft F1 |
| **RQ3** | Do retrieved three-shot examples improve accuracy — including on unseen templates? | Δ EX zero-shot → 3-shot (similar / fixed) |
| **RQ4** | SQL-specialised vs. general models at similar scale? | Head-to-head across all workloads |
| **RQ5** | What are the latency, memory, and energy costs per correct answer? | Wall-clock s, peak VRAM, joules / correct |

## Models

All models served as **Q4\_K\_M GGUF** through Ollama with F16 KV cache, flash attention, single-request concurrency.

| Model | Params | Type | VRAM | Source |
|-------|-------:|------|-----:|--------|
| Prem-1B-SQL | 1.35 B | SQL | 1.7 GB | HF → GGUF |
| Qwen2.5-Coder-1.5B | 1.5 B | Code | 1.2 GB | Ollama |
| Qwen2.5-Coder-3B | 3.1 B | Code | 2.2 GB | Ollama |
| XiYanSQL-3B | 3.1 B | SQL | 2.2 GB | HF → GGUF |
| Llama-3.2-3B | 3.2 B | General | 2.6 GB | Ollama |
| Qwen3-4B | 4.0 B | General | 3.2 GB | Ollama |

## Hardware

| | |
|---|---|
| **Board** | NVIDIA Jetson Orin Nano Super Developer Kit |
| **Memory** | 8 GB LPDDR5 (shared CPU / GPU) |
| **JetPack** | 6.2.3 · L4T R36.5.2 |
| **Power mode** | MAXN\_SUPER · `jetson_clocks` |
| **Serving** | Ollama 0.30.11 · llama.cpp `ecdd29e8` |
| **Energy sensor** | On-board VDD\_IN via tegrastats @ 100 Hz |

## Benchmarks

| Benchmark | Items | Metrics | Design |
|-----------|------:|---------|--------|
| NYC Taxi | 449 eval / 744 pool | Strict EX | 26 templates, 4 held-out folds |
| Spider dev | 1,034 | EX, TS, EM | Standard development split |
| BIRD Mini-Dev v1 | 500 | EX, Soft F1 | SQLite subset |

## Repository Structure

```
.
├── reproduce.py                # One command: verify → analyze → check → build
├── requirements.txt
│
├── experiment/
│   ├── PLAN.md                 # Full experimental protocol
│   └── code/
│       └── score.py            # Scoring pipeline (NYC · Spider · BIRD)
│
├── analysis/
│   └── analyze.py              # Raw scores → paper numbers + tables
│
├── tools/
│   ├── common.py               # Analysis helpers (CIs, formatting)
│   ├── render.py               # Fill {{placeholders}} → final paper
│   ├── fingerprint.py          # SHA-256 integrity verification
│   └── verify_citations.py     # Bibliography metadata checker
│
├── paper/
│   ├── paper.md                # Manuscript source
│   ├── references.bib          # Bibliography
│   └── numbers.lock.json       # Author-verified values
│
├── results/raw/                # Read-only experimental outputs
│   └── c9-main-20261017-s1/
│       ├── acc/                # .jsonl per model × condition
│       ├── env/                # Environment snapshots + model hashes
│       ├── logs/               # Session logs, flags, events
│       └── power/              # VDD_IN power traces (100 Hz)
│
├── data/
│   ├── taxi.db                 # NYC taxi SQLite database
│   ├── questions/              # Question sets (generated + curated)
│   └── benchmarks/             # Spider dev + BIRD Mini-Dev v1
│
└── build/                      # Generated (gitignored)
    ├── numbers.json
    ├── tables/
    └── figures/
```

## Reproduction

### Quick start — verify existing results

```bash
git clone https://github.com/biolababs/edgeslm-text-to-sql.git
cd edgeslm-text-to-sql
pip install -r requirements.txt

# Download benchmark evaluators (one-time, ~1.5 GB)
python setup.py --download-benchmarks

# Verify everything: fingerprints → analysis → number check → citations → paper
python reproduce.py
```

`reproduce.py` runs five stages:

1. **Fingerprint** — SHA-256 hashes confirm raw data is unmodified
2. **Analyze** — regenerate all numbers and tables from scored outputs
3. **Check** — compare every paper value against `numbers.lock.json`
4. **Citations** — verify bibliography metadata against online sources
5. **Render** — fill placeholders and build the final document

### Rerun the full experiment

Requires a Jetson Orin Nano Super with JetPack 6.2+ and Ollama 0.30+.
See [`experiment/PLAN.md`](experiment/PLAN.md) for the complete protocol.

```bash
python experiment/code/run.py --config experiment/config/main.json
python experiment/code/score.py \
    --run results/raw/<your-run-id> \
    --cfg experiment/config/score_paths.json
python analysis/analyze.py --run <your-run-id>
```

## Companion Study

Controlled precision comparison (F16 / Q8 / Q4 from the same source weights) and on-device LoRA fine-tuning are reserved for a separate companion paper using the same infrastructure and raw data.

## Citation

```bibtex
@misc{biolababs2026texttosql,
  title   = {Text-to-SQL on an 8\,GB Edge Device: Accuracy, Generalisation,
             and Energy Cost of Small Open-Weight Models},
  author  = {Biola Adeniyi},
  year    = {2026},
  url     = {https://github.com/biolababs/edgeslm-text-to-sql}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
