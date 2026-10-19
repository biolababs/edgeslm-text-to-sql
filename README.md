# Text-to-SQL on an 8 GB Edge Device

**Accuracy, Generalisation, and Energy Cost of Small Open-Weight Models on the NVIDIA Jetson Orin Nano Super**

This repository contains the reproducibility package for our study evaluating six small open-weight text-to-SQL models (1B to 4B parameters, 4-bit quantised) on an NVIDIA Jetson Orin Nano Super with 8 GB of memory.

## Key Findings

- **Three-shot prompting with retrieved similar examples** raises NYC taxi accuracy from 9--31% (zero-shot) to 70--83% (S-similar) across all models
- **SQL-specialised models dominate published benchmarks** but do not consistently outperform general models on a novel database
- **Prem-1B-SQL** achieves 84.5% Spider EX from only 1.35B parameters, yet scores lowest zero-shot on an unfamiliar schema
- All measurements include **per-query energy** from on-board VDD\_IN power sensors

## Models Evaluated

| Model | Parameters | Type | VRAM (Q4\_K\_M) |
|---|---|---|---|
| Prem-1B-SQL | 1.35B | SQL-specialised | 1.7 GB |
| Qwen2.5-Coder-1.5B | 1.5B | Code | 1.2 GB |
| Qwen2.5-Coder-3B | 3.1B | Code | 2.2 GB |
| XiYanSQL-3B | 3.1B | SQL-specialised | 2.2 GB |
| Llama-3.2-3B | 3.2B | General | 2.6 GB |
| Qwen3-4B | 4.0B | General | 3.2 GB |

## Repository Structure

```
.
├── README.md
├── LICENSE
├── reproduce.py              # One command to rebuild and verify everything
├── requirements.txt          # Python dependencies
├── experiment/
│   ├── PLAN.md               # Full experimental protocol
│   └── code/
│       └── score.py          # Scoring pipeline (NYC, Spider, BIRD evaluators)
├── analysis/
│   └── analyze.py            # Generates all paper numbers and tables
├── tools/
│   ├── common.py             # Shared analysis helpers
│   ├── render.py             # Fills paper placeholders from numbers.json
│   ├── fingerprint.py        # SHA-256 data integrity verification
│   └── verify_citations.py   # Citation metadata checker
├── paper/
│   ├── paper.md              # Manuscript source (Markdown)
│   ├── references.bib        # Bibliography
│   └── numbers.lock.json     # Locked paper values (after author sign-off)
├── results/
│   └── raw/                  # Raw experimental outputs (read-only)
│       └── c9-main-20261017-s1/
│           ├── acc/           # Model outputs (.jsonl per condition)
│           ├── env/           # Environment snapshots
│           ├── logs/          # Session logs, flags, events
│           └── power/         # VDD_IN power traces
├── data/
│   ├── taxi.db               # NYC taxi SQLite database (seeded sample)
│   ├── questions/            # Generated + curated question sets
│   └── benchmarks/           # Spider dev + BIRD Mini-Dev v1 (see setup)
└── build/                    # Generated (gitignored except locked values)
    ├── numbers.json
    ├── scores/
    ├── tables/
    └── figures/
```

## Reproduction

### Requirements

- Python 3.12+
- NVIDIA Jetson Orin Nano Super (for re-running experiments; not needed for analysis verification)
- ~2 GB disk for raw data and benchmark evaluators

### Quick Start

```bash
# Clone and install
git clone https://github.com/biolababs/edgeslm-text-to-sql.git
cd edgeslm-text-to-sql
pip install -r requirements.txt

# Download benchmark evaluators (Spider test-suite, BIRD evaluator)
python setup.py --download-benchmarks

# Verify data integrity and reproduce all paper numbers
python reproduce.py
```

### What `reproduce.py` Does

1. **Verifies raw data** -- SHA-256 fingerprints confirm nothing was modified
2. **Reruns analysis** -- `analyze.py` regenerates all numbers and tables from scored outputs
3. **Confirms numbers** -- every value in the paper is checked against `numbers.lock.json`
4. **Verifies citations** -- bibliography metadata is checked against online sources
5. **Builds paper** -- fills placeholders and renders the final document

### Re-running Experiments

To re-run the full experiment on your own Jetson:

```bash
# Requires: Jetson Orin Nano Super, JetPack 6.2+, Ollama 0.30+
# See experiment/PLAN.md for the complete protocol
python experiment/code/run.py --config experiment/config/main.json
python experiment/code/score.py --run results/raw/<your-run-id> --cfg experiment/config/score_paths.json
python analysis/analyze.py --run <your-run-id>
```

## Hardware

- **Device:** NVIDIA Jetson Orin Nano Super Developer Kit
- **Memory:** 8 GB LPDDR5 (shared CPU/GPU)
- **JetPack:** 6.2.3, L4T R36.5.2
- **Power mode:** MAXN\_SUPER with `jetson_clocks`
- **Serving:** Ollama 0.30.11, llama.cpp (commit ecdd29e8)
- **Energy sensor:** On-board VDD\_IN via tegrastats (100 Hz)

## Benchmarks

- **NYC Taxi:** 449 evaluation items, 26 templates, 4 held-out folds
- **Spider:** 1,034 development items (EX, TS, EM)
- **BIRD Mini-Dev v1:** 500 items, SQLite (EX, Soft F1)

## Citation

```bibtex
@misc{biolababs2026edgeslm,
  title   = {Text-to-SQL on an 8\,GB Edge Device: Accuracy, Generalisation,
             and Energy Cost of Small Open-Weight Models},
  author  = {[Authors to be supplied]},
  year    = {2026},
  url     = {https://github.com/biolababs/edgeslm-text-to-sql}
}
```

## License

[To be determined -- see LICENSE file]

## Companion Study

Controlled precision comparison (F16/Q8/Q4) and on-device LoRA fine-tuning results will be published in a separate companion paper using the same experimental infrastructure.
