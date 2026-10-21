<p align="center">
  <h1 align="center">Edge SLM Text-to-SQL</h1>
  <p align="center">
    <b>Accuracy and Energy Trade-offs of Small Text-to-SQL Models<br>on an 8 GB Edge Device — A Reproducible Measurement Study</b>
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

**Example retrieval is the effect that survives scrutiny.** Supplying three
retrieved demonstrations raises strict execution accuracy on the taxi database
from 8.5–31.2 % (zero-shot) to 70.2–82.7 %. The effect holds for every model
under an exact McNemar test on paired items (p from 3 × 10⁻¹⁵ to 6 × 10⁻⁷²).

**About half that gain is template recall, not schema reasoning.** Each item is
measured twice: once with its own query template present in the example pool,
once with it withheld. Accuracy falls from 89.8–99.8 % to 50.6–68.6 % — a gap of
28.3–48.8 pp, decisive for every model (exact McNemar, p from 6 × 10⁻³¹ to
2 × 10⁻⁶⁶; every cluster-bootstrap interval excludes zero).

Decomposing the three-shot gain: examples supply roughly half the benefit
regardless of template match, and template exposure supplies the other half
(mean 49.3 %, range 37.1–59.7 % across the six models). Retrieval gains measured
on template-generated question sets will therefore overstate performance on
genuinely novel queries — the curated question set, which shares less vocabulary
with the pool, gains far less.

**Differences *between models* are not resolvable at this sample size.** The two
leading models differ by 1.4 pp, 95 % CI [−4.4, 7.1].

**Energy per correct answer reorders the models.** The cheapest correct answer
costs 42.4 J (QwenCoder-1.5B); the highest-scoring model costs 94.5 J.

**Benchmark standing did not predict local ordering here.** Prem-1B-SQL leads
Spider (84.5 %) but reaches 14.9 % zero-shot on the taxi database, while
QwenCoder-3B is fourth on Spider and first on the taxi set. With six models the
rank correlation is weak and not significant (ρ = 0.26, p = 0.62).

> ⚠️ **A cautionary result worth reusing.** Comparing corrected-annotation BIRD
> scores on a 369-item subset against v1 scores on the full 500 items appears to
> show gains of up to +9.4 pp. On a *fixed* item set, correcting the reference
> SQL in fact *lowers* accuracy for five of six models. The apparent gain was a
> change of denominator. See `bird_sensitivity` in the paper.

All numbers above are generated from the raw data in this repository;
`reproduce.py` regenerates and checks every one.

## Research Questions

| # | Question | Evaluation |
|:-:|---------|-----------|
| **RQ1** | How accurately do the models answer taxi queries, and how much does accuracy change when a query's own template is withheld from the example pool? | Strict EX on 449 items measured under both exposures, paired within items; template-cluster bootstrap intervals and exact McNemar |
| **RQ2** | How do the same artifacts score on published benchmarks, and how sensitive are those scores to annotation quality? | Spider EX / TS / EM; BIRD EX / Soft F1; paired corrected-gold comparison on a fixed item set |
| **RQ3** | Do three-shot examples improve accuracy, does retrieval beat a fixed set, and how much of the gain is template recall rather than schema reasoning? | Δ EX zero-shot → 3-shot, exact McNemar on paired items, gain decomposed by exposure |
| **RQ4** | Do SQL-specialised models outperform general and code models at similar scale? | Head-to-head across all workloads, reported as observed differences with intervals |
| **RQ5** | What are the latency, memory, and energy costs per correct answer? | Wall-clock s, peak VRAM, joules / correct |

> **Scope.** This is a measurement study. Where the design cannot resolve a
> difference, the paper reports that rather than ranking the models.

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
| NYC Taxi | 449 eval / 744 pool | Strict EX | 26 templates, 4 pool folds; each item measured with its template present and withheld |
| Spider dev | 1,034 | EX, TS, EM | Standard development split |
| BIRD Mini-Dev v1 | 500 | EX, Soft F1 | SQLite subset |

## Repository Structure

The reusable tooling is kept free of anything specific to this study, so it can
be pointed at another paper without editing it. Everything that knows about
*these* six models and *this* taxi database lives in `analysis/`, `experiment/`
and a single config file.

```
.
├── reproduce.py                  # verify -> analyse -> check -> citations -> build
├── requirements.txt              # requirements.lock.txt pins exact versions
│
├── tools/                        # REUSABLE - no knowledge of this study
│   ├── common.py                 # numbers, tables, figures, Wilson intervals
│   ├── render.py                 # fills {{placeholders}} from numbers.json
│   ├── fingerprint.py            # SHA-256 manifest over results/raw
│   ├── verify_citations.py       # Crossref + arXiv lookup, with retry
│   └── audit_claims.py           # guards ranking/quantifier claims in prose
│
├── analysis/                     # THIS STUDY
│   ├── analyze.py                # scored outputs -> numbers + tables
│   ├── stats_tests.py            # cluster bootstrap, McNemar, Holm, energy
│   └── figures.py                # the seven figures
│
├── paper/
│   ├── paper.md                  # manuscript source; every result is a placeholder
│   ├── references.bib            # 22 references, all verified
│   ├── audit.config.json         # models + metrics the claim auditor ranks on
│   └── claims.reviewed.json      # 206 prose claims, verified and pinned by hash
│
├── experiment/
│   ├── PLAN.md                   # the registered protocol
│   ├── code/                     # runner, scorer, test-mode and env scripts
│   ├── config/                   # generation settings, model files, score paths
│   ├── data/                     # question sets and schema
│   ├── curated/                  # the 92-item curated set and its provenance
│   ├── prompts/                  # prompt templates
│   └── requests/                 # the exact prompt sent for every request
│
├── results/
│   ├── MANIFEST.sha256           # hashes for every raw file
│   ├── RUN_REPORT.md             # what happened during the run
│   └── raw/c9-main-20261017-s1/
│       ├── acc/                  # raw per-query outputs, one .jsonl per model x condition
│       ├── env/                  # environment snapshots and model hashes
│       ├── logs/                 # session logs, flags, events
│       └── power/                # VDD_IN trace (gzipped) and GPU clocks
│
└── build/scores/                 # scored outputs - INPUT to the analysis, see below
```

**Not included, by design:** model weights (`.gguf`), LoRA adapters from
fine-tuning dry runs that no reported result depends on, and the Spider
test-suite and BIRD evaluator databases. None are ours to redistribute, and all
are obtainable from their original sources.

## Reproduction

### Verify the published results

```bash
git clone https://github.com/biolababs/edgeslm-text-to-sql.git
cd edgeslm-text-to-sql
pip install -r requirements.lock.txt

# The power trace ships compressed; the manifest hashes the uncompressed file
gunzip -k results/raw/c9-main-20261017-s1/power/session_s1-block_B3.log.gz
gunzip -k results/raw/c9-main-20261017-s1/env/ollama_journal_s1.txt.gz

python reproduce.py
```

`reproduce.py` runs five stages:

1. **Fingerprint** — SHA-256 hashes confirm the raw data is unmodified
2. **Analyse** — regenerate every number, table and figure from the scored outputs
3. **Check** — compare each value against `paper/numbers.lock.json`
4. **Citations** — re-resolve the bibliography against Crossref and arXiv
5. **Render** — fill the placeholders and build the document

This path needs no database and no benchmark evaluator: it consumes
`build/scores/`, which is why those scored outputs ship with the archive rather
than being regenerated.

### Re-score from the raw outputs

Scoring executes predicted SQL, so it needs the taxi database and the benchmark
evaluators and their databases. Obtain Spider and BIRD from their own
distributions, then:

```bash
python experiment/code/score.py     --run results/raw/c9-main-20261017-s1     --cfg experiment/config/score_paths.json
```

### Re-run the experiment

Requires a Jetson Orin Nano Super with JetPack 6.2+ and Ollama 0.30+, and the
model artifacts named in `experiment/config/modelfiles/`. Set `JETSON_HOST` to
`user@host` for the board. See [`experiment/PLAN.md`](experiment/PLAN.md) for the
protocol, including the test-mode procedure and the power-sampling setup.

### Checking the prose, not just the numbers

Every reported value in the manuscript is substituted from `numbers.json`, so no
number can drift. Claims *about* those numbers are hand-written, and three
review rounds found errors in exactly that layer. `tools/audit_claims.py` finds
every ranking, superlative, ordinal and universal-quantifier claim, prints the
true ordering beside it, and fails until each is verified and pinned by hash:

```bash
python tools/audit_claims.py              # fails on any unreviewed claim
python tools/audit_claims.py --rankings   # ground-truth ordering per metric
```

## Companion Study

Controlled precision comparison (F16 / Q8 / Q4 from the same source weights) and on-device LoRA fine-tuning are reserved for a separate companion paper using the same infrastructure and raw data.

## Citation

```bibtex
@misc{biolababs2026texttosql,
  title   = {Accuracy and Energy Trade-offs of Small Text-to-SQL Models on an
             8\,GB Edge Device: A Reproducible Measurement Study},
  author  = {Abiola Babsalaam},
  year    = {2026},
  url     = {https://github.com/biolababs/edgeslm-text-to-sql}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
