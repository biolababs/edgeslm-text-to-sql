# Text-to-SQL on an 8 GB Edge Device

> Accuracy, generalisation, and energy cost of small open-weight models on the NVIDIA Jetson Orin Nano Super

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Can a small language model translate natural-language questions into correct SQL on an 8 GB edge board — and what does each correct answer cost in energy?

This repository is the reproducibility package for a study that evaluates **six open-weight text-to-SQL models (1 B – 4 B parameters, 4-bit quantised)** on an NVIDIA Jetson Orin Nano Super. The evaluation covers three complementary workloads: a single-domain taxi database with held-out query templates, the Spider development set, and BIRD Mini-Dev v1. Every accuracy figure is paired with per-query latency and module-input energy drawn from the board's on-board VDD_IN sensor.

## Research Questions

| # | Question |
|---|---------|
| RQ1 | How accurately do the models answer taxi queries, and how does performance change when query templates are withheld from the example pool? |
| RQ2 | How do the same artifacts score on Spider (EX, TS, EM) and BIRD Mini-Dev (EX, Soft F1)? |
| RQ3 | Do retrieved three-shot examples improve accuracy — including on held-out templates? |
| RQ4 | How do SQL-specialised models compare with general and code-oriented models at similar scale? |
| RQ5 | What are the latency, memory, and energy costs per correct answer on the edge board? |

## Models

| Model | Parameters | Type | VRAM (Q4\_K\_M) |
|-------|-----------|------|----------------|
| Prem-1B-SQL | 1.35 B | SQL-specialised | 1.7 GB |
| Qwen2.5-Coder-1.5B | 1.5 B | Code | 1.2 GB |
| Qwen2.5-Coder-3B | 3.1 B | Code | 2.2 GB |
| XiYanSQL-3B | 3.1 B | SQL-specialised | 2.2 GB |
| Llama-3.2-3B | 3.2 B | General | 2.6 GB |
| Qwen3-4B | 4.0 B | General | 3.2 GB |

All models are served as Q4\_K\_M GGUF artifacts through Ollama with an F16 KV cache and flash attention, single-request concurrency.

## Hardware

| Component | Detail |
|-----------|--------|
| Board | NVIDIA Jetson Orin Nano Super Developer Kit |
| Memory | 8 GB LPDDR5 (shared CPU/GPU) |
| JetPack | 6.2.3, L4T R36.5.2 |
| Power mode | MAXN\_SUPER, `jetson_clocks` |
| Serving | Ollama 0.30.11, llama.cpp `ecdd29e8` |
| Energy sensor | On-board VDD\_IN via tegrastats @ 100 Hz |

## Repository Structure

```
├── reproduce.py              # Verify data, rerun analysis, rebuild the paper
├── requirements.txt
├── experiment/
│   ├── PLAN.md               # Full experimental protocol
│   └── code/
│       └── score.py          # Scoring pipeline (NYC, Spider, BIRD evaluators)
├── analysis/
│   └── analyze.py            # Generate all paper numbers and tables
├── tools/
│   ├── common.py             # Shared analysis helpers
│   ├── render.py             # Fill paper placeholders from numbers.json
│   ├── fingerprint.py        # SHA-256 data integrity verification
│   └── verify_citations.py   # Citation metadata checker
├── paper/
│   ├── paper.md              # Manuscript source (Markdown with placeholders)
│   ├── references.bib        # Bibliography
│   └── numbers.lock.json     # Locked values (author-verified)
├── results/
│   └── raw/                  # Read-only experimental outputs
│       └── c9-main-20261017-s1/
│           ├── acc/          # Model outputs (.jsonl per condition)
│           ├── env/          # Environment snapshots and model hashes
│           ├── logs/         # Session logs, flags, events
│           └── power/        # VDD_IN power traces (100 Hz)
├── data/
│   ├── taxi.db               # NYC taxi SQLite database
│   ├── questions/            # Generated + curated question sets
│   └── benchmarks/           # Spider dev + BIRD Mini-Dev v1
└── build/                    # Generated (gitignored)
```

## Reproduction

### Prerequisites

- Python 3.12+
- ~2 GB disk for raw data and benchmark evaluators
- An NVIDIA Jetson Orin Nano Super is needed only to rerun experiments; analysis verification runs on any machine

### Verify and Rebuild

```bash
git clone https://github.com/biolababs/edgeslm-text-to-sql.git
cd edgeslm-text-to-sql
pip install -r requirements.txt

# Download benchmark evaluators (Spider test-suite, BIRD evaluator)
python setup.py --download-benchmarks

# Full verification: fingerprints → analysis → number check → citations → paper
python reproduce.py
```

### Rerun Experiments

```bash
# Requires: Jetson Orin Nano Super, JetPack 6.2+, Ollama 0.30+
# See experiment/PLAN.md for the complete protocol
python experiment/code/run.py --config experiment/config/main.json
python experiment/code/score.py \
    --run results/raw/<your-run-id> \
    --cfg experiment/config/score_paths.json
python analysis/analyze.py --run <your-run-id>
```

## Benchmarks

| Benchmark | Items | Metrics | Notes |
|-----------|-------|---------|-------|
| NYC Taxi | 449 eval, 744 pool | EX (strict) | 26 templates, 4 held-out folds |
| Spider dev | 1,034 | EX, TS, EM | Standard development split |
| BIRD Mini-Dev v1 | 500 | EX, Soft F1 | SQLite subset |

## Companion Study

Controlled precision comparison (F16 / Q8 / Q4 from the same source weights) and on-device LoRA fine-tuning are reserved for a separate companion paper using the same experimental infrastructure and raw data.

## Citation

```bibtex
@misc{biolababs2026texttosql,
  title   = {Text-to-SQL on an 8\,GB Edge Device: Accuracy, Generalisation,
             and Energy Cost of Small Open-Weight Models},
  author  = {[Authors]},
  year    = {2026},
  url     = {https://github.com/biolababs/edgeslm-text-to-sql}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
