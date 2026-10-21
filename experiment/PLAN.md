# Step 4 · Experimental protocol (C9, text-to-SQL with small open models on an edge box)

Status: draft for team sign-off, 2026-10-16. Nothing in this plan has been run yet. Inputs: MOTIVATION.md, LITERATURE.md (sections d and e), QUESTIONS.md (RQ1 to RQ7, H1 to H6), `experiment/prior-code/`, and the exploratory logs in `results/raw/exploratory-2026-10-13/`. I used those logs only to estimate run times and to find problems; none of their numbers are evidence. Section 15 lists the decisions that need sign-off. Until the team signs off, the defaults below are proposals.

Abbreviations: **EX** = execution accuracy (the predicted query's result equals the gold result). **TS** = test-suite accuracy (EX over Spider's distilled test-suite databases, Zhong et al. 2020). **EM** = Spider exact-set match. **Q4** = GGUF Q4_K_M. **S-sim / S-fixed** = the two three-shot example strategies (section 5.3). **Fold** = a held-out template group (section 4.2).

---

## 0. What I found while preparing this plan (it changes the design)

These come from read-only inspection of the Jetson on 2026-10-16 and from re-reading the exploratory logs.

1. **The exploratory LoadGen results are invalid for 3 of 4 models.** `run_all.log` lines 743, 768 and 793 show `python loadgen_bench.py --model qwen2.5-coder:1.5b --out runs/perf/<other model>`. All four `result.json` files report `"model": "qwen2.5-coder:1.5b"`, with near-identical numbers (p90 2.54 s, 31.5 J/query). The `run_all.sh` kept in `prior-code` passes `--model $m`, so the script that actually ran was not the one preserved. The llama-bench files are fine: they point to each model's own blob.
2. **A second LLM service was resident throughout the exploratory pass.** The `jetson-agent` stack (supervisor PID 1186, started at boot 3 days 7 hours before inspection) keeps `llama-server` running with LFM2.5-1.2B-Instruct (Q4_0, 16k context) loaded: about 1 GB RSS plus its GPU allocation. It also runs `opencode web`, a Node proxy (ports 8000 and 8787) and a Python workflow server (8790). Its logs show no requests since boot, so it was probably idle. It still held memory, and it is the likely source of the 0.7 to 0.9 GB of zram swap seen at idle.
3. **The three-shot prompt caused runaway generations.** In `qwen2.5-coder_1.5b.nyc3shot.jsonl`, 73 of 514 outputs hit the 384-token cap. The model read the three example `Question:/SQL:` pairs as questions to answer ("Let's walk through each question..."). The prompt format must change (section 5).
4. **The LoRA run trained successfully but never produced a model.** Training finished (174 steps, 2,346 s, loss 0.0001 by step 60, i.e. heavy overfitting). The process was then killed at the in-PyTorch `merge_and_unload()` (out of memory: 7.5 of 7.6 GB RAM plus 3.3 GB swap). The follow-up `ft_merge.sh` failed with `libcudss.so.0: cannot open shared object file` only because it did not export `LD_LIBRARY_PATH`. **cuDSS 0.7.1 is already on the board at `~/bench/cudss/lib`.**
5. **Spider TS cannot be computed yet.** `~/bench/test-suite-sql-eval/database/` contains only `readme.txt`. The distilled databases (Google Drive id `1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w`) were never downloaded, so the exploratory "Spider" numbers are EX on the original dev databases plus EM.
6. **Ollama is not at defaults.** Ollama 0.30.11 runs with `OLLAMA_HOST=0.0.0.0`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_NUM_PARALLEL=1` and `OLLAMA_LLM_LIBRARY=cuda_jetpack6`. A quantised KV cache confounds the precision control (RQ5), and a 0.0.0.0 listener lets other LAN clients send requests mid-run.
7. **jetson_clocks is still applied** from the exploratory pass: CPUs at 1.728 GHz min = max, GPU at 1.020 GHz min = max, EMC override at 3.199 GHz. The board's default DVFS state before that is not on record.
8. **Ollama's Q4_K_M is not Qwen's official Q4_K_M.** The Ollama blob for `qwen2.5-coder:1.5b` is 986,048,576 bytes. Qwen's official `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` is 1,117,320,768 bytes. Qwen publishes no F16 GGUF. A clean precision ladder therefore needs one converter and one source for all rungs (section 2.2).
9. **Licences differ from what MOTIVATION implies.** `qwen2.5-coder:3b` is under the **Qwen Research License** (non-commercial). The other Qwen models are Apache-2.0, and Llama 3.2 has its own community licence. This matters for the decision table.
10. **Generation defaults differ by model.** For example, the qwen3 Modelfile sets temperature 0.7, top_k 20 and top_p 0.8, and the Ollama default repeat penalty applies when a Modelfile does not set one. Greedy decoding with an uncontrolled repeat penalty is not pure greedy, so all sampling options will be set explicitly (section 5.5).
11. **Prior code assumed a sudo password** (`$SUDO_PW` in `run_all.sh`). Passwordless sudo is now enabled, and no password is ever stored or passed.

Platform recorded at inspection: Jetson Orin Nano Engineering Reference Developer Kit Super, L4T R36.5.2 (`nvidia-l4t-core 36.5.2-20260716114719`), JetPack 6.2.3+b81, kernel 5.15.199-tegra, CUDA 12.6 (driver 540.5.0), nvpmodel MAXN_SUPER (mode 2), RAM 7,607 MB, zram 6 x 634 MB plus a 16 GB `/swapfile`, NVMe 915 GB (789 GB free), Python 3.10.12, SQLite 3.37.2, llama.cpp `ecdd29e8` (2026-10-13), mlcommons_loadgen 6.0.17, and in `~/bench/ftenv`: torch 2.11.0, transformers 4.57.6, peft 0.21.2, accelerate 1.15.0, gguf 0.19.0. Idle VDD_IN was about 6.6 W at tj about 53 °C, with clocks locked and the agent resident.

---

## 1. Objectives mapped to the research questions

| RQ | Objective | Blocks (section 11) | Primary measurement |
|---|---|---|---|
| RQ1 | Accuracy of 4-bit 1.5B to 4B open models on the NYC taxi database, on seen vs held-out query types | B3, B4 | Strict EX on the NYC template set (449 items, 26 templates, 4 folds) and the curated set (≥60). Wilson and template-cluster bootstrap CIs, per template and per fold. |
| RQ2 | Position against published benchmarks | B3, B6 | Spider dev (1,034): EX (original DB), **TS**, EM (reference). BIRD Mini-Dev **v1, SQLite, 500**: EX, Soft F1. Corrected Mini-Dev (Arcwise-Plat-SQL, 498) EX as a sensitivity check. Every number labelled by metric, set and precision. |
| RQ3 | Does three-shot prompting help, and on seen or only unseen types? | B3, B4 | Paired EX: 0-shot vs 3-shot S-sim (seen and unseen) and 3-shot S-fixed, using exact McNemar on the same items. Within-item seen vs unseen contrast. |
| RQ4 | SQL-specialised small model vs code and general models | B3, B5 | XiYanSQL-QwenCoder-3B-2504 (primary) and Prem-1B-SQL (secondary) vs the four general and code models, plus the paired XiYan-3B vs Qwen2.5-Coder-3B base comparison. EX, TS, EX per GB. |
| RQ5 | Cost of 4-bit quantisation | B6 | Qwen2.5-Coder-1.5B at F16, Q8_0 and Q4_K_M, all converted by us from the same official weights with one pinned llama.cpp, on all four sets. Paired differences. |
| RQ6 | Cost per answer on the Orin Nano Super | B8 (and per-query energy from every accuracy run) | LoadGen SingleStream (≥1,024 queries, ≥600 s): p50, p90 (early-stopping estimate), p99, QPS. VDD_IN energy per query (total and above idle), **energy per correct answer**, TTFT proxy, decode tok/s, peak RAM, swap delta, max tj, GPU clock floor. llama-bench pp512/tg128 as a reference. |
| RQ7 | On-box LoRA: cost and effect on unseen types | B7 | Training wall time and VDD_IN energy for 4 fold models. EX before and after, seen and unseen, 0-shot and 3-shot S-sim, paired with the base model on the same items. |

Hypotheses are tested as set out in section 8.4.

---

## 2. Models

### 2.1 Models table

All are open-weight. For every file, the SHA-256 is recorded in `experiment/config/models.lock.json` at preparation time and checked on the Jetson before each block. "Ollama library" files are the blobs already on the Jetson; I have recorded their SHA-256 (it is the blob name). Hugging Face revisions are pinned to the commit SHAs listed. Sizes are file sizes in bytes.

| # | Name (tag in runs) | Params | Quant | Exact file | SHA-256 | Licence | Source | Why included |
|---|---|---|---|---|---|---|---|---|
| M1 | Qwen2.5-Coder-1.5B-Instruct (`qc15-q4-ollama`) | 1.5B | Q4_K_M | Ollama blob, 986,048,576 B | `29d8c98fa6b098e200069bfb88b9508dc3e85586d20cba59f8dda9a808165104` | Apache-2.0 | Ollama library `qwen2.5-coder:1.5b` (ID d7372fd82851) | Exploratory model. Same model as Renney 2026. Smallest code model. LoRA base. |
| M2 | Qwen2.5-Coder-3B-Instruct (`qc3-q4-ollama`) | 3.1B | Q4_K_M | Ollama blob, 1,929,903,072 B | `4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba` | **Qwen Research License (non-commercial)** | `qwen2.5-coder:3b` (ID f72c60cabf62) | Code model, mid size. Base of XiYan-3B, so it gives a paired specialisation contrast. |
| M3 | Llama 3.2 3B Instruct (`llama32-3b-q4-ollama`) | 3.2B | Q4_K_M | Ollama blob, 2,019,377,376 B | `dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff` | Llama 3.2 Community License | `llama3.2:3b` (ID a80c4f17acd5) | General instruct model. Same model as Renney 2026. |
| M4 | Qwen3-4B-Instruct-2507 (`qwen3-4b-q4-ollama`) | 4.0B | Q4_K_M | Ollama blob, 2,497,280,480 B | `85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9` | Apache-2.0 | `qwen3:4b-instruct-2507-q4_K_M` (ID 0edcdef34593) | Largest general model that fits comfortably. Non-thinking variant. |
| M5 | XiYanSQL-QwenCoder-3B-2504 (`xiyan3b-q4-ours`) | 3.1B | Q4_K_M | Our conversion: HF safetensors → F16 GGUF → Q4_K_M | recorded at prep | Apache-2.0 per card (**derived from Qwen2.5-Coder-3B, see note**) | HF `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` @ `b883e58ed83f74bab037d6a7b90c4b8706d357d7` | **SQL-specialised (fix 3).** Card reports BIRD dev 52.09 (DDL schema) and 55.08 (M-Schema). Same architecture as M2, so specialisation is isolated. |
| M6 | Prem-1B-SQL (`prem1b-q4-ours`) | 1.35B | Q4_K_M | Our conversion as for M5 | recorded at prep | Apache-2.0 tag / MIT in card text (**conflict**) | HF `prem-research/prem-1B-SQL` @ `44dd7fcf9227af4efed936bf29323c61bf66aad1` | Second SQL-specialised model, 1B tier (card: BIRD dev 46, test 51.5). Gives the specialised arm two sizes. |
| P1 | Qwen2.5-Coder-1.5B-Instruct (`qc15-f16-ours`) | 1.5B | F16 | Our conversion from HF safetensors | recorded at prep | Apache-2.0 | HF `Qwen/Qwen2.5-Coder-1.5B-Instruct` @ `2e1fd397ee46e1388853d2af2c993145b0f1098a` | **Precision control (fix 4)**, top rung. About 3.1 GB, so it fits. |
| P2 | same (`qc15-q8-ours`) | 1.5B | Q8_0 | `llama-quantize` from P1 | recorded at prep | Apache-2.0 | as P1 | Precision control, middle rung. |
| P3 | same (`qc15-q4-ours`) | 1.5B | Q4_K_M | `llama-quantize` from P1 | recorded at prep | Apache-2.0 | as P1 | Precision control, bottom rung. Same converter as P1 and P2, unlike M1. |
| L1 to L4 | M1 plus a LoRA adapter for fold *f* (`qc15-lora-f{1..4}`) | 1.5B + 18.5M | Q4_K_M base + F16 LoRA GGUF applied at runtime | Ollama Modelfile `FROM` the M1 blob, `ADAPTER lora-f{f}.gguf` | adapters recorded at creation | Apache-2.0 | Trained on the Jetson (B7) | **RQ7.** Using the identical base file as M1 means base vs LoRA differ only by the adapter. |
| (opt) O1 | OmniSQL-7B (`omnisql7b-q4-ours`) | 7.6B | Q4_K_M (about 4.7 GB) | Our conversion | recorded at prep | Apache-2.0 | HF `seeklhy/OmniSQL-7B` @ `af4eed67f561bbeea555c017dae4b38b93bac2eb` | **Optional, needs sign-off.** A specialised upper bound that still fits in 8 GB once the agent stack is stopped. Outside the 1.5B to 4B frame of RQ1. |

Notes on the specialised models (fix 3), checked 2026-10-16 through the Hugging Face API and model cards:
- **XiYanSQL-QwenCoder-3B-2504** is tagged Apache-2.0, is not gated, and is distributed as safetensors only. Community GGUFs exist (`mradermacher/XiYanSQL-QwenCoder-3B-2504-GGUF`, Q4_K_M 1,929,903,424 B, sha256 `de8d9ca0...`). Its base, Qwen2.5-Coder-3B, is under the Qwen Research License. The derivative's Apache tag may therefore not override the base terms. The team should confirm before the paper recommends it for commercial use.
- **Prem-1B-SQL** has metadata Apache-2.0, card body MIT, and a base reported as DeepSeek-Coder-1.3B. Its inference prompt is defined by the PremSQL library, not the card. The pilot checks that it produces SQL under our common prompt (section 10).
- **Excluded:** FINER-SQL-3B/4B-BIRD (no licence declared on the card, which in effect means all rights reserved; community GGUFs exist). SLM-SQL (CC-BY-NC-4.0). CodeS-1B/3B (`.bin` weights only, a base model with no chat template, and it expects a specific prompt with sample values; StarCoder OpenRAIL-M lineage). OmniSQL is listed only as an optional 7B model.

### 2.2 How each file is produced and checked

- **Ollama library models (M1 to M4):** already on the Jetson. They are not pulled again. Before each block, `sha256sum` of the blob must equal the table value, and `ollama show --modelfile` is saved.
- **Our conversions (M5, M6, P1 to P3, O1):** these are built on the PC, not the Jetson. The PC downloads the pinned HF revision and runs `convert_hf_to_gguf.py` from llama.cpp commit `ecdd29e8b2076bb85da9e9ca7ee3d84264375bd1` (the same commit as the Jetson build) to F16. It then runs `llama-quantize` from that commit to Q8_0 and Q4_K_M, with no imatrix, matching how the Ollama library builds its quants. If `llama-quantize` cannot be built on the PC, the fallback is the existing Jetson binary at the same commit, run outside test mode as preparation; which one was used is recorded. The GGUFs are stored outside the project in `C:\Users\biola\paper\models-cache\c9\`, with hashes in `models.lock.json`. They are copied to the Jetson with `scp` and registered with `ollama create <tag> -f Modelfile`. The Modelfile `TEMPLATE` comes from the GGUF's embedded `tokenizer.chat_template`, and the pilot checks the rendered template.
- **Cross-check:** our Q4_K_M of XiYan is compared byte-size and tensor-type-wise with mradermacher's file, as a sanity check only.
- **Model memory:** after warm-up, `ollama ps` must show `100% GPU` for every model. Any CPU split invalidates the block.

---

## 3. Datasets and splits

| Set | Version and source | Size | Use | Licence |
|---|---|---|---|---|
| NYC taxi DB | TLC yellow taxi 2024-01 parquet plus zone lookup. `build_db.py --month 2024-01 --rows 200000` (seed 0). | 200,000 trips, 3 tables | All NYC sets | TLC open data |
| NYC template instances | `gen_questions.py --per-template 60 --seed 0` over 26 templates (`templates.py`) | 1,193 instances (22 to 60 per template) | Eval set, shot pool, LoRA training | ours |
| **NYC eval set** | `make_folds.py` (new, seed 0): per template, the first m_t = min(20, ⌊n_t / 2⌋) instances of a seeded shuffle | **449 items, 26 templates** | 0-shot, 3-shot, LoRA evaluation | ours |
| **NYC pool** | the remaining instances | 744 | 3-shot example pool; LoRA training (fold-filtered) | ours |
| **NYC curated set** | "curated (AI-drafted)", section 3.3. The 15 exploratory seed items are kept as a flagged subset. | target ≥60 accepted (draft about 80) | 0-shot, 3-shot | ours |
| Spider dev | Spider 1.0 `spider_data.zip`, sha256 `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b` (already on the Jetson) | 1,034 questions, 20 DBs | 0-shot | CC BY-SA 4.0 |
| Spider test suites | test-suite-sql-eval distilled DBs, Google Drive `1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w`. Evaluator commit `e97acc546ecbee8fa27fa8dbf025ef61493a876c`. | DBs for 20 dev schemas | TS scoring only (on the PC) | Apache-2.0 (repo) |
| BIRD Mini-Dev **v1** | original `minidev.zip` from the BIRD OSS bucket, sha256 `cc48ba16838204e4e214512030cb572eeb5f7bcdd999bae4b9b6ff12ec13b92f` (downloaded 2026-10-13). `mini_dev_sqlite.json`, **SQLite**. Evaluator `bird-bench/mini_dev` @ `abd11b6db92a1c9f809b32f7564c7c71b34d67f0`. | **500** (148 simple / 250 moderate / 102 challenging), 11 DBs | 0-shot | CC BY-SA 4.0 |
| **Corrected Mini-Dev** (fix 2) | Jin et al. 2026, `uiuc-kang-lab/text_to_sql_benchmarks` @ `fe766045c55b6875a43b30e9ac7683df5582f8cf`, file `data/arcwise_plat_sql_only_with_diff.json` (Arcwise-Plat-SQL: corrected gold SQL only; questions and evidence unchanged). Checked: 498 items, 11 DBs, SQLite syntax, keyed by `question_id`. | 498 | **Re-scoring of the v1 predictions, no extra inference** | CC BY-SA 4.0 |

Decisions on BIRD variants:
- **Arcwise-Plat-SQL is the sensitivity check.** Its questions and evidence are identical to v1, so the same predictions are re-scored against corrected gold SQL. That isolates annotation error from model behaviour at zero Jetson cost. Difficulty labels are joined from v1 by `question_id`.
- **Arcwise-Plat (full) is not used.** It also changes questions, evidence and schemas (`data/schemas`), so it would need new inference and modified databases.
- **Mini-Dev V2 (780, 2025-07-22) is not used.** It adds 270 instances, including CRUD and JSON operations on 18 new databases in LiveSQLBench format, which needs state-based evaluation. No small-model results are published on it, so it adds no comparability. The paper will say so.
- **The Hugging Face `birdsql/bird_mini_dev` release** is the "canonical" version for new users. Its equality with our v1 zip is checked at prep (item-by-item diff of question, evidence and SQL). Any difference is reported, and v1 from the zip remains the scored version.

### 3.1 Reproducing the NYC data

1. On the PC, run `build_db.py` and `gen_questions.py` exactly as in the exploratory pass (seed 0, `--per-template 60`, default holdout).
2. The regenerated `train`, `test_seen` and `test_unseen` JSONL must be **byte-identical** to `results/raw/exploratory-2026-10-13/taxi_sql/data/questions/*.jsonl`, and `schema.sql` identical. If they are not, stop and report: TLC may have republished the parquet. Record the parquet SHA-256.
3. `make_folds.py` pools all 1,193 instances, then assigns `eval`/`pool` and a fold to each template (section 4.2). It writes `nyc_eval.jsonl`, `nyc_pool.jsonl` and `folds.json`.
4. Gold answers are computed on the PC (Python 3.12 sqlite3; version recorded). Scoring also runs on the PC, so gold and scoring use the same SQLite build. An oracle run (gold SQL as predictions) must give 100.0% strict EX on every NYC file.

### 3.2 Sizes and planning precision (95% CI half-width at p = 0.5)

NYC eval 449: ±4.6 pp per condition, but only 26 templates, so the template-cluster bootstrap is the honest interval. Per fold, unseen is 105 to 121 items (±9 pp). Curated 60: ±12.6 pp. Spider 1,034: ±3.0 pp. BIRD 500: ±4.4 pp. Paired McNemar on 449 items with about 25% discordance detects a difference of about 6 to 7 pp at 80% power (α = 0.05).

### 3.3 Curated (AI-drafted) set: procedure

The team decision fixes the label: "curated (AI-drafted)", never "hand-written".

1. **Coverage grid.** 10 categories x 8 drafts = 80 drafts: filtering/counting; aggregation; grouping; ranking/top-k; ratios and percentages; time (hour, day of week, date ranges); multi-join (pickup and dropoff zones); nested/subquery; anti-join/set operations; NULL and edge semantics (zero fares, cash tips not recorded). Phrasing must avoid template wording.
2. **Drafting.** An AI assistant drafts the question first and the SQL second. The model name, version, date and drafting prompt are recorded in `experiment/curated/DRAFTING.md`.
3. **Two independent gold queries per question.** A second SQL is written in a separate session without seeing the first. Both must return identical results under `eval_exec.py` normalisation. On disagreement, a team reviewer decides and documents the reason, or the item is rewritten or dropped.
4. **Automatic checks.** Each item must execute in under 10 s, return non-empty and non-degenerate results, pass the `rank_sql`-style tie check at any `LIMIT` cut, and have the `ordered` flag set when order matters.
5. **Human check.** A named team member reads every accepted item (question, SQL, answer) and signs `experiment/curated/REVIEW.md`.
6. **Structure tags.** Each item gets a structure tag: `template-like` if its SQL skeleton, with literals and identifiers masked, matches any of the 26 templates, else `novel`. Results are reported both overall and by tag.
7. **Freeze.** `curated.jsonl` is hashed into `experiment/config/data.lock.json` before any model sees it. After the freeze, items are never edited. A later-found error is marked excluded, with the reason, and results are reported with and without it.
8. **Size rule.** At least 60 accepted. If fewer than 60 survive, more are drafted before freezing, not after.
9. **Seed items.** The 15 exploratory seed items go through the same checks. They stay flagged as `seed`, because the team has already seen model outputs on them. Results are reported with and without them.

---

## 4. Conditions

### 4.1 Conditions matrix

Item counts assume 70 curated items (C = 70). "✓" means a full run; "-" means not run.

| Model | NYC 0-shot (449 + C) | NYC 3-shot S-sim (898 + C) | NYC 3-shot S-fixed (449 + C) | Spider dev 0-shot (1,034) | BIRD Mini-Dev v1 0-shot (500) (+ Arcwise re-score) | LoadGen 0-shot | LoadGen 3-shot S-sim (opt) | llama-bench |
|---|---|---|---|---|---|---|---|---|
| M1 qc15-q4-ollama | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ x3 | ✓ | ✓ |
| M2 qc3-q4-ollama | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| M3 llama32-3b-q4-ollama | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| M4 qwen3-4b-q4-ollama | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ x3 | ✓ | ✓ |
| M5 xiyan3b-q4-ours | ✓ (+ native prompt) | ✓ | ✓ | ✓ | ✓ (+ native prompt) | ✓ | ✓ | ✓ |
| M6 prem1b-q4-ours | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| P1 qc15-f16-ours | ✓ | ✓ | - | ✓ | ✓ | ✓ | - | ✓ |
| P2 qc15-q8-ours | ✓ | ✓ | - | ✓ | ✓ | ✓ | - | ✓ |
| P3 qc15-q4-ours | ✓ | ✓ | - | ✓ | ✓ | - (llama-bench only) | - | ✓ |
| L1 to L4 qc15-lora-f*f* | ✓ on fold test set (about 225) + C | ✓ on fold test set (about 225) | - | opt (L1 only) | - | ✓ (L1) | - | - |
| (opt) O1 omnisql7b | ✓ | ✓ | - | ✓ | ✓ | ✓ | - | ✓ |

Extra conditions:
- **Clocks-off sensitivity (optional):** LoadGen 0-shot for M1 and M4 with jetson_clocks restored to DVFS, still in MAXN_SUPER.
- **Template-recall baseline (PC only, no Jetson):** "nearest-neighbour copy". For each eval item, execute the SQL of its top-1 S-sim neighbour from the fold pool, in the item's seen and unseen folds. This shows how much of the seen benefit is copying and confirms that unseen items cannot be solved by recall (Finegan-Dollak).
- **Spider and BIRD get zero-shot only (item 9 of the brief).** Justification:
  1. RQ3 asks about in-database examples on the deployment database. Cross-domain train-set examples answer a different question.
  2. The published comparison points for our models are zero-shot or greedy single-sample. The CodeS few-shot table is TS on base models with its own prompt, so it is not a like-for-like anchor.
  3. A three-shot condition would add about 1.5 h per model (about 14 h in total) for a result that informs no listed decision.
  The paper will state that Spider and BIRD were run zero-shot only.

### 4.2 Held-out template folds (fix 6)

Four folds partition all 26 templates, so every template is "unseen" in exactly one fold and "seen" in the other three. That rotation balances template difficulty across the seen/unseen comparison, which a single held-out set cannot do. Fold 1 is the original construct-novel set, which keeps the stronger "new SQL construct" test. Folds 2 to 4 hold out templates whose constructs also appear in training. They test new *template structure* (Finegan-Dollak query split), not new constructs. The paper reports fold 1 separately from folds 2 to 4.

| Fold | Held-out templates (eval items) | Eval items | Train/shot pool for this fold |
|---|---|---|---|
| F1 (novel construct) | highest_avg_fare_zone_min_pickups (HAVING) 20, cross_borough_trips_hour (<>) 12, avg_speed_borough (derived) 20, busiest_date_borough (date()) 20, zones_without_pickups (NOT IN) 20, above_avg_fare_zone (scalar subquery) 20 | 112 | 532 |
| F2 | count_trips_borough 14, avg_tip_by_borough 20, busiest_pickup_zone_hour 12, pct_payment_borough 20, count_borough_to_borough 20, max_total_date 15, distinct_dropoffs_from_zone 20 | 121 | 546 |
| F3 | revenue_borough_dow 14, longest_trip_per_payment_borough 20, peak_hour_zone 20, avg_tolls_borough_to_borough 20, count_passengers_miles 14, avg_tip_pct_hour_credit 12, airport_dropoffs_from_borough 11 | 111 | 576 |
| F4 | avg_fare_zone 20, avg_duration_borough 20, top_k_dropoff_zones_from_borough 17, top_tip_borough 20, trips_per_payment_borough 14, avg_passengers_miles 14 | 105 | 578 |

Folds 2 to 4 are balanced by construct family: each has at least one argmax (ORDER BY ... LIMIT) template, one GROUP BY-list template, one payment-join template and one two-join template. A known limitation goes in the paper: F1's "derived per-row quantity" (avg_speed) shares the minutes expression with avg_duration_borough (F4). When F1 is held out, that expression is therefore present in the pool.

**Paired seen/unseen design.** For 3-shot S-sim and LoRA, the run for fold *g* evaluates:
- the eval items of fold *g* as **unseen** (their templates are absent from pool *g*); and
- the eval items of fold *g−1* (cyclic, so F1 uses F4) as **seen** (their templates are in pool *g*; the items themselves are never in any pool).

Every eval item is therefore scored exactly once seen and once unseen, so seen vs unseen is a **within-item paired comparison**. That gives 898 3-shot S-sim queries per model. Zero-shot has no examples, so "seen/unseen" does not exist for it. Each item is run once at zero-shot, and zero-shot accuracy by fold group is reported only as a template-difficulty baseline.

---

## 5. Prompts and generation settings

All prompts are built by one function in `experiment/code/prompts.py`. Its rendered output for one example of every condition is saved in `experiment/prompts/rendered_examples.txt` during the pilot. Messages go to Ollama `/api/chat`.

### 5.1 Common prompt (all models, all sets, unless "native" is stated)

System message, verbatim:

```
You are a SQLite expert. Given the database schema below, write one SQLite query that answers the user's question.
Return only the SQL query, with no explanation.

{schema}
```

`{schema}` is the CREATE TABLE statements in code representation, with no sample rows:
- **NYC:** `data/schema.sql`, including its column comments.
- **Spider and BIRD:** the DDL stored in each SQLite file (`prep_public.py ddl()`, unchanged).

User message (zero-shot), verbatim:

```
{hint_line}Question: {question}
```

`{hint_line}` is `Hint: {evidence}\n` for BIRD (the official with-evidence setting) and empty otherwise.

Compared with the exploratory prompt, the schema moves to a system message and examples become chat turns. This fixes the runaway-generation problem (section 0, item 3). The exploratory results are therefore not comparable with the new ones and are not reused as evidence.

### 5.2 Three-shot prompt

The system message is the same. Then come three prior turns, (user `Question: {q_i}`, assistant `{sql_i}`), and then the target user turn. Examples are ordered by increasing similarity, so the most similar example sits next to the question. Assistant turns contain the bare gold SQL with no fences, which matches the requested output format.

### 5.3 Example selection

- **S-sim (primary).** The top-3 pool questions by TF-IDF cosine similarity on question text. The tokeniser is lower-case `[a-z0-9]+`, IDF is computed over the fold pool, and ties are broken by item id. The implementation is stdlib-only and deterministic. Retrieval uses the fold pool defined in section 4.2. For curated items, the full pool (744) is used, since they have no template. The share of seen items whose top-1 shot shares their template is reported, as a property of the retrieval.
- **S-fixed (secondary).** Three examples fixed for every question in a fold: from fold *f*'s pool, one instance from each of three distinct templates, drawn with `random.Random(0)`. Each eval item is run once, in its own held-out fold (unseen). Curated items use fold 1's fixed set. This answers whether a generic handful of in-database examples helps without retrieval.

### 5.4 Native prompt for XiYan (sensitivity, RQ4 fairness)

The `nl2sqlite_template_cn` template is copied verbatim from the model card at revision `b883e58` into `experiment/prompts/xiyan_nl2sqlite_cn.txt` and hashed. It is sent as a single user message with `dialect="SQLite"`, `db_schema` = the same DDL as above (we do not build M-Schema), and `evidence` = the BIRD hint or empty. It runs on NYC 0-shot and BIRD only. The card reports BIRD dev 52.09 with DDL. Prem-1B-SQL uses the common prompt. If the pilot shows ≥20% non-SQL outputs for it, the PremSQL default template is adopted for Prem-1B-SQL alone, before the full run, and the change is documented.

### 5.5 Generation settings (identical for every model and condition)

| Setting | Value | Note |
|---|---|---|
| Decoding | greedy: `temperature 0`, `top_k 1`, `top_p 1.0`, `min_p 0`, `repeat_penalty 1.0`, `presence/frequency_penalty 0` | All set explicitly to override Modelfile defaults (section 0, item 10). XiYan's card suggests temperature 0.1 sampling. Greedy is used for comparability and is noted in the paper. |
| Seed | `seed 0` | |
| Context | `num_ctx 4096` | Largest prompt seen is about 2,171 tokens (BIRD). The runner asserts `prompt_eval_count + num_predict ≤ 4096`; items exceeding it are flagged, not silently truncated. |
| Max new tokens | `num_predict 384` | Gold SQL is under 150 tokens. Cap hits are reported as "truncated". |
| Thinking | `"think": false` | qwen3-instruct-2507 is non-thinking. Any `<think>` block is stripped. |
| Keep-alive | `keep_alive "30m"` per request; models are unloaded explicitly between blocks | |
| Ollama server (test mode) | `OLLAMA_FLASH_ATTENTION=1`, **`OLLAMA_KV_CACHE_TYPE=f16`** (proposed; deployed value is q8_0), `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_HOST=127.0.0.1:11434` | Section 9 and decision D7 |
| Warm-up | 5 untimed requests on pool items after each model load; outputs discarded | |
| Item order | file order for accuracy runs; LoadGen samples with fixed seeds | |

**SQL extraction** (fixed before the run; the rule used is recorded per item):
1. Strip `<think>…</think>`.
2. If a fenced block exists, take the first ```` ```sql ````/```` ```sqlite ````/```` ``` ```` block.
3. Otherwise take text from the first `SELECT` or `WITH` keyword to the end of the first statement (the first `;` outside quotes), or to the end of the text.
4. If nothing is found, the item is "no SQL".

---

## 6. Metrics and how each is computed

### 6.1 Accuracy

- **NYC strict EX (headline).** `eval_exec.py` as in prior code: row multiset equality (or ordered equality when `ordered`), numbers within abs 0.01 or rel 1e-4, strings compared case-folded and trimmed, read-only connection, 10 s timeout, at most 10,000 rows.
- **NYC lenient EX (secondary, our own definition).** Extra projected columns are allowed if some column subset matches. It is always named "lenient EX (ours)" and is never compared with Soft F1 or TS.
- **Error taxonomy (NYC, Spider, BIRD).** Each item falls into one of: no SQL extracted / truncated (cap hit) / execution error (syntax, unknown table or column, timeout) / executes but wrong / correct.
- **Spider:**
  - **EX (original DB):** `evaluation.py --etype exec --db spider_data/database`.
  - **TS:** `--etype exec --db test-suite database/` with the distilled DBs, default flags (no `--plug_value`, since we predict values; `--keep_distinct` off).
  - **EM (reference only):** `--etype match`.
  - All three are reported by hardness (easy/medium/hard/extra).
- **BIRD Mini-Dev v1:** EX via `evaluation_ex.py` and Soft F1 via `evaluation_f1.py` (SQLite, `--meta_time_out 30`), reported by difficulty. **R-VES is not reported**, because timing evaluation on the scoring machine says nothing about the edge box.
- **Corrected Mini-Dev:** EX of the same predictions against Arcwise-Plat-SQL gold, on the 498 items, using the same evaluator. We report the EX change per model and Kendall's τ between model rankings on v1 and corrected.
- **Labelling rule.** Every reported number carries (metric, set, version, precision, prompt), for example "TS, Spider dev, Q4_K_M, 0-shot".

### 6.2 Cost (per run and per query)

- **Per-query timing (all accuracy and LoadGen runs).**
  - Wall time from client send to response, with `time.time_ns()` timestamps recorded at both ends.
  - Ollama `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`, `load_duration`.
  - **TTFT proxy** = `load_duration + prompt_eval_duration` (server-side).
  - **Prefill tok/s** = `prompt_eval_count / prompt_eval_duration`.
  - **Decode tok/s** = `eval_count / eval_duration`.
- **Power trace.**
  - `tegrastats --interval 100` piped through `power_log.py`, which prefixes each line with `time.time_ns()`.
  - tegrastats is the established tool; the prefix only adds sub-second timestamps.
  - Logged fields: VDD_IN, VDD_CPU_GPU_CV, VDD_SOC (mW), RAM, SWAP, per-sensor temperatures, GR3D load.
  - A 1 s sysfs sampler logs the GPU devfreq `cur_freq` (`/sys/class/devfreq/17000000.gpu/cur_freq`), the CPU `scaling_cur_freq`, and the INA3221 `curr*_crit_alarm` flags.
- **Energy per query (J).** Trapezoidal integral of VDD_IN over each query's [start, end] window.
  - **Total** energy is the integral as measured.
  - **Above idle** = total − P_idle x duration, where P_idle is the median VDD_IN of the nearest board-idle baseline (no model loaded, same clocks, samplers running).
  - **Model-loaded idle** power is also recorded and reported.
- **LoadGen run energy.** ∫VDD_IN over [first issue, last completion] divided by queries; total and above idle.
- **Energy per correct answer (J).** Σ energy over a run's items divided by the number of strictly correct items in that run.
  - For LoadGen, each issued sample is mapped to its item, and correctness comes from the accuracy run with the same prompt. Greedy determinism is verified (section 10).
  - Reported for NYC (eval and curated), with item-bootstrap CIs.
- **Latency (LoadGen SingleStream).** p50, p90, p99 and mean from `mlperf_log_summary.txt`; the early-stopping p90 estimate; QPS = completed queries / duration; "Result is VALID".
- **Memory and thermal.**
  - Peak RAM (MB), and peak minus pre-load idle RAM.
  - Swap used delta (MB, zram plus file) from start to end of a block. Any increase over 50 MB flags the block.
  - Max tj (°C).
  - Minimum GPU clock under load. With clocks locked, anything below 1,020 MHz or any `crit_alarm` = 1 flags throttling.
  - Model size from `ollama ps`.
- **Measurement boundary.** VDD_IN is module input power from the on-board INA3221. It excludes the carrier board, NVMe, fan and USB. It is not a wall meter and **not MLPerf Power** (fix 5); the wording is "following the LoadGen SingleStream scenario; power from on-board sensors (VDD_IN)".
- **Training (B7).** Wall time (s), steps, examples, tokens, mean VDD_IN, energy total and above idle (kJ and Wh), peak RAM, swap delta, max tj, `torch.cuda.max_memory_allocated`, and adapter size.
- **llama-bench.** `-p 512 -n 128 -r 5 -ngl 99 -o json` per GGUF, at the same commit as the exploratory build. Reported as pp512 and tg128 tok/s (mean ± sd), which is comparable in kind to Renney's tok/s.

### 6.3 LoadGen SingleStream settings (fix 5)

| Setting | Value |
|---|---|
| `scenario` / `mode` | SingleStream / PerformanceOnly |
| `min_query_count` | 1024 |
| `min_duration_ms` | 600000 |
| `target_latency_percentile` | 90 (default) |
| `single_stream_expected_latency_ns` | pilot mean for that model |
| `qsl_rng_seed`, `sample_index_rng_seed`, `schedule_rng_seed` | fixed constants, recorded in the run's `user.conf` |
| QSL | NYC eval plus curated prompts (0-shot or S-sim), `performance_sample_count` = QSL size |

Minimums checked:
- **Reddi et al. 2020:** SingleStream needs at least 1,024 queries and at least 60 s.
- **Current MLCommons inference rules (checked 2026-10-16):** minimum duration 600 s for SingleStream. There is no fixed minimum query count (early stopping governs validity), and the metric is the 90th-percentile early-stopping latency estimate. There are no LLM-specific SingleStream token-latency rules for edge.

**Our settings meet both the 1,024-query and the 600 s minimums.** Deviations stated in the paper: the models are not MLPerf reference models; there is no accuracy mode or compliance tests; and power comes from on-board sensors, not a SPEC-certified meter. So this is "following the LoadGen SingleStream scenario", not an MLPerf result.

Fixes to `loadgen_bench.py`:
- The result's `model` is taken from the server response and asserted equal to the requested tag.
- `ollama ps` and the blob SHA-256 are saved with each run.
- Per-sample (index, start, end, tokens, output text) is logged.
- Energy is integrated with timestamps, not mean power x wall time.

---

## 7. Fine-tuning protocol (RQ7, fix 7)

- **Base:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` @ `2e1fd397`. The safetensors already in `~/bench/hf/qwc15` are checked against the HF LFS SHA-256 and re-downloaded to the run workspace if they differ.
- **Data:** fold *f*'s pool (532 to 578 examples; held-out templates excluded). The training prompt is exactly the **zero-shot** chat prompt of section 5.1, rendered with the tokenizer's chat template. Loss is computed on the SQL answer tokens only (the `answer_loss` from prior code is kept).
- **Hyperparameters (pre-registered):** LoRA r = 16, α = 32, dropout 0.05, on all attention and MLP projections. AdamW, lr 2e-4, cosine schedule with 5% warm-up, weight decay 0, batch 1 with gradient accumulation 8, max length 1,024, bf16, gradient checkpointing, **1 epoch**, seed 0. The exploratory pass used 2 epochs; its loss reached 0.001 by step 30 (about 240 examples), so 1 epoch is enough and halves cost (decision D11).
- **Merge failure fix.** No in-PyTorch merge (it ran out of memory), and no merge on the Jetson at all.
  1. `finetune.py` saves only the adapter (`adapter_model.safetensors`, about 74 MB).
  2. The adapter is copied back to the PC and converted there with `convert_lora_to_gguf.py` (llama.cpp `ecdd29e8`, CPU torch in `experiment/envs/convert`) to an F16 LoRA GGUF.
  3. It is registered on the Jetson as `FROM <M1 blob>` + `ADAPTER lora-f{f}.gguf`.
  This takes `libcudss` out of the critical path. On the Jetson, every `ftenv` process still sets `LD_LIBRARY_PATH=$HOME/bench/cudss/lib`, because the Jetson torch wheel needs it at import; that is the root cause of the exploratory failure.
- **Fallback, if Ollama's `ADAPTER` path misbehaves in the pilot** (output identical to base on training items, or a load error): use `llama-export-lora` (F16 base + LoRA) → `llama-quantize Q4_K_M`, on the PC. The accuracy comparison is then against P3-style "our Q4" of the same base, and the change is documented.
- **Optional merged deployment variant:** the merged Q4 GGUF (made as in the fallback) for L1 only, as an accuracy sanity check plus llama-bench, showing whether merging changes accuracy or speed.
- **Measurements:** training runs under `power_log.py`, with the run-level metrics from section 6.2. The idle baseline is taken immediately before each training run.
- **Evaluation:**
  - L*f* runs on fold *f*'s test set: the fold *f* items (unseen) and the fold *f−1* items (seen), at 0-shot and 3-shot S-sim (pool *f*), plus curated at 0-shot.
  - Every item gets a LoRA-seen and a LoRA-unseen score, paired with M1's 0-shot and 3-shot results on the same items.
  - The LoRA plus 3-shot condition tests the DAIL-SQL finding that in-context ability is lost after fine-tuning.
  - Optional: L1 on Spider dev, to check how much general SQL ability the on-box adaptation costs.
- **Minimum if time is short:** folds F1 and F2 (decision D11).

---

## 8. Statistical analysis plan

Everything is computed on the PC by `experiment/code/stats.py` from raw outputs. The bootstrap uses B = 10,000 resamples with seed 20261016 and percentile intervals.

### 8.1 Single proportions
- 95% **Wilson** interval (item level) for every EX and TS proportion. Soft F1 is not a proportion, so it gets the database-cluster bootstrap only.
- For NYC, also a **template-cluster bootstrap**: resample the 26 templates with replacement, then all items within each chosen template.
- Curated: item bootstrap.
- Spider: database-cluster bootstrap (20 DBs). BIRD: database-cluster bootstrap (11 DBs).
- Both intervals are reported. The cluster interval is the one used for conclusions.

### 8.2 Paired comparisons (same items)
- **Exact two-sided McNemar** on discordant counts (b, c), plus the paired difference with a paired cluster-bootstrap CI.
- **Families**, each corrected with **Holm–Bonferroni**:
  - **RQ3:** per model (6 Q4 models + P1, P2 + P3), 0-shot vs S-sim-unseen, 0-shot vs S-sim-seen, S-sim-seen vs S-sim-unseen, and 0-shot vs S-fixed.
  - **RQ4:** M5 vs M2 (paired base), M5 vs best non-specialised model, M6 vs M1.
  - **RQ5:** P1 vs P2, P2 vs P3, P3 vs M1 (Ollama vs our Q4).
  - **RQ7:** per seen/unseen x 0/3-shot, L vs M1.
- Model rankings: pairwise McNemar on NYC 0-shot and Spider TS, reported as a matrix, with no correction claimed beyond Holm within the family.

### 8.3 Cost statistics
- Latency percentiles come from LoadGen.
- For the repeated runs (M1 and M4, 3 x each), we report run-to-run CV of p90 and of energy per query. If CV exceeds 5%, the team decides whether all configurations get extra repeats.
- Energy per correct answer gets an item-bootstrap CI that resamples (energy_i, correct_i) pairs.

### 8.4 Hypotheses: operationalisation and decision rules (need sign-off, D4)

| H | Test | Supported if |
|---|---|---|
| H1 | Within-item seen vs unseen at 3-shot S-sim, and at LoRA 0-shot, per model | Seen − unseen ≥ 10 pp and the cluster-bootstrap 95% CI excludes 0, **for every model** (F1 and F2 to F4 also shown separately). Zero-shot has no seen/unseen; its fold-group difference is reported only as template difficulty. |
| H2 | Gain over 0-shot on seen vs on unseen, same items: (acc3_seen − acc0) − (acc3_unseen − acc0) | The difference is > 0 with CI excluding 0. Under the paired design this equals the 3-shot seen − unseen contrast of H1. The two are reported once, with H1 also tested under LoRA. The unseen gain alone (acc3_unseen − acc0) answers RQ3's "do they help on unseen types". |
| H3 | EX per GB of file (NYC 0-shot strict EX and Spider TS, divided by file size in GB = 10⁹ B) | A specialised model (M5 or M6) has the highest point value on both, and its bootstrap CI for the difference with the next model excludes 0 |
| H4 | P2 (Q8_0) − P3 (Q4_K_M), paired, on NYC 0-shot, NYC 3-shot, Spider TS, BIRD EX | Upper 95% bound of the drop ≤ 3 pp on each set |
| H5 | NYC energy per correct answer (LoadGen 0-shot energy/query ÷ NYC 0-shot EX; also from the accuracy runs, and at 3-shot) | The model with the highest NYC EX does not have the lowest J/correct, **and** the lowest J/correct belongs to a 3B-class model (M2, M3 or M5), with CIs separating it from the larger and the smaller alternatives |
| H6 | L vs M1 on the same items, 0-shot | Seen gain ≥ 15 pp (CI lower bound > 5 pp) **and** unseen gain ≤ one third of the seen gain |

Multiplicity: H1 to H6 are confirmatory, and everything else is exploratory and labelled so in the paper.

---

## 9. Test mode and environment recording

Scripts: `experiment/code/testmode.sh {enter|verify|exit}` and `experiment/code/env_record.sh <run_dir>`. Both are copied to the Jetson per run and run via `ssh -o BatchMode=yes $JETSON_HOST`. No passwords are used anywhere; sudo is passwordless. Neither script reads `~/jetson-agent/config/api-keys.txt`. The agent's own `health.sh` reads it internally and is called as-is.

### 9.0 Before entering (go/no-go)
1. `who`, `ps -eo user,pcpu,rss,etime,args --sort=-pcpu | head -30`, `ollama ps` (must be empty), and `tegrastats` for 30 s (GR3D_FREQ 0% throughout).
2. Check that no established connections exist on ports 11434, 8000 or 8787 (`ss -tnp state established`).
3. Check the last request time in `~/jetson-agent/logs/server.log`.
4. Confirm with the team that agent downtime is acceptable for the session window.

If anything is in use, **stop and report**; do not run alongside it.

### 9.1 Enter
1. **Snapshot** (saved to `results/raw/<run-id>/env/pre/` and to `~/c9-testmode-state/` for restore):
   - `systemctl list-units --type=service --state=running`
   - `systemctl list-timers --all`
   - `crontab -l`
   - `systemctl show ollama -p Environment`
   - `nvpmodel -q`
   - `jetson_clocks --show`
   - `swapon --show`
   - `ss -tlnp`
   - jetson-agent PIDs
2. **jetson-agent stack:** run `~/jetson-agent/scripts/stop.sh`. It signals supervisor 1186, whose cleanup trap stops llama-server (both processes), opencode, the proxy and the workflow server. Verify after 60 s: no `llama-server|opencode|proxy/server.js|workflow/server.py` processes, and ports 8000, 8001, 8787, 8788, 8790 and 58699 are closed. It restarts only from `@reboot`, so **no reboot during a session**.
3. **Display manager:** `sudo systemctl stop gdm` (decision D8). This frees about 100 to 200 MB and removes compositor GPU wake-ups.
4. **Periodic jobs.** These are stopped, never disabled:
   - `sudo systemctl stop` on `trafficedge-ip-reporter.timer apt-daily.timer apt-daily-upgrade.timer fwupd-refresh.timer man-db.timer motd-news.timer update-notifier-download.timer update-notifier-motd.timer anacron.timer e2scrub_all.timer fstrim.timer dpkg-db-backup.timer logrotate.timer systemd-tmpfiles-clean.timer`
   - `sudo systemctl stop cron` (pauses the 6-hourly `ec2_healthcheck.sh`)
   - `sudo systemctl stop fwupd packagekit`
   - `sudo snap refresh --hold=96h`
   - `ssh-tunnel.service`, `docker`/`containerd` (no containers running) and cups are left running but recorded. Docker must show `docker ps` empty.
5. **Ollama.**
   - Add the drop-in `/etc/systemd/system/ollama.service.d/zz-c9-testmode.conf`, setting `OLLAMA_HOST=127.0.0.1:11434` and `OLLAMA_KV_CACHE_TYPE=f16` (if D7 is approved). All other variables are unchanged.
   - Run `daemon-reload` and `restart ollama`.
   - Verify `systemctl show ollama -p Environment` and that `ss -tlnp` shows `127.0.0.1:11434` only.
6. **Power and clocks.**
   - `sudo nvpmodel -m 2`, then verify `NV Power Mode: MAXN_SUPER`.
   - `sudo jetson_clocks`, then verify with `jetson_clocks --show`: CPU min = max = 1,728 MHz, GPU min = max = 1,020 MHz, EMC 3,199 MHz.
   - The fan stays on `nvfancontrol` dynamic control, recorded.
7. **Swap and caches.**
   - `sudo swapoff -a && sudo swapon -a` flushes the stale ~0.7 GB in zram. It needs enough free RAM, which is checked first.
   - Then `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`.

### 9.2 Verify idle (must pass before any measurement)
Run 180 s of `power_log.py` together with `top -b -d 5`. All of these must hold:
- GR3D_FREQ 0% in ≥ 99% of samples.
- Mean CPU < 5%.
- VDD_IN sd < 0.2 W.
- No process other than the samplers above 1% CPU on average.
- `ollama ps` empty.
- Listening sockets limited to sshd, rpcbind, systemd-resolved, cupsd and ollama (loopback).
- Swap used < 50 MB.

The output is saved as `env/idle_verify.txt`. On failure: stop, restore, report.

**Idle baseline.** After verification, record 300 s of board-idle power (no model loaded). This is repeated at the start of each block and before each LoRA training run. **Model-loaded idle**: 120 s after warm-up, per model.

### 9.3 Environment record (per run, into `results/raw/<run-id>/env/`)
- `/etc/nv_tegra_release`, `/proc/device-tree/model`, `dpkg -l nvidia-jetpack nvidia-l4t-core`, `uname -a`
- `nvpmodel -q`, `jetson_clocks --show`, `free -m`, `swapon --show`, `df -h`, `uptime`
- `ollama -v`, `systemctl show ollama -p Environment`, `ollama list`, `ollama show --modelfile` for each model, and `sha256sum` of each GGUF used
- `nvidia-smi` header; llama.cpp `git log -1`
- `python3 -V`; `pip freeze` of `~/bench/ftenv`; sqlite3 version
- LoadGen version; evaluator commits; dataset hashes
- Running services and timers; `ps` top 30; thermal zones; INA3221 labels
- Plus `run_meta.json`: run id, block, start and end UTC, git-style hash of `experiment/` code, condition list, and **jetson_clocks state and power mode recorded per run** (fix 8)

### 9.4 Exit (restore and confirm)
1. Unload all models (`ollama stop` each). Remove the drop-in, `daemon-reload`, `restart ollama`. Verify the Environment equals the pre-snapshot and that it listens on `*:11434`.
2. Start the stopped timers, `cron`, `fwupd` and `packagekit`. Run `snap refresh --unhold`. Then `sudo systemctl start gdm`.
3. Clocks: leave in the state found at entry (currently locked; decision D6 asks whether the team wants DVFS restored, which needs a reboot because no pre-lock `--store` file exists).
4. Start the agent with `nohup ~/jetson-agent/scripts/start.sh` (as trafficedge01). Wait up to 120 s, then run `~/jetson-agent/scripts/health.sh` (exit 0 required). Verify that ports 8000, 8001, 8787, 8788 and 8790 are listening and that the liquid-general llama-server is loaded.
5. Diff running services, timers and listening ports against the pre-snapshot. The diff (only transient units allowed) is saved to `env/restore_diff.txt`.
6. Copy back results (`scp -r`) to `results/raw/<run-id>/`. Compare SHA-256 manifests on both sides, then delete the run workspace `~/c9/<run-id>/` on the Jetson. Benchmark data in `~/bench` stays as test-bed fixtures.

Sessions: test mode is entered and exited per session (at most about 24 h). Each session starts with 9.0 to 9.3 and ends with 9.4, so the agent is down only during sessions.

### 9.5 Thermal, cool-down and swap rules (fix 8)
- **Power mode MAXN_SUPER and jetson_clocks ON for every main run.**
  - Justification: locked CPU, GPU and EMC clocks remove DVFS ramp variance. This makes latency and energy comparisons between models repeatable and fair. It is also the state NVIDIA uses for Jetson LLM benchmarks and the state the board is in now.
  - Cost: idle power is higher, so absolute energy is a performance-locked figure. We report above-idle energy alongside total, and run a clocks-off sensitivity check on two models (optional block) to quantify the difference for DVFS deployments.
- **Cool-down between models and between LoadGen runs.**
  - Unload, then wait until tj ≤ T_idle + 2 °C, where T_idle is the session's idle-baseline median tj.
  - Minimum 3 min, maximum 20 min. If 20 min is reached, log it and continue, with a flag.
- **Throttling flag.** GPU `cur_freq` below 1,020 MHz under load, any INA3221 `crit_alarm`, or tj ≥ 95 °C. Flagged blocks are reported, and LoadGen runs with a flag are repeated under a new run id.
- **Swap flag.** Used swap (zram plus file) rising by more than 50 MB during a block. Flagged LoadGen runs are repeated after a swap flush. Accuracy runs are kept, with the flag.
- **Ambient.** If the team has a thermometer near the board, record room temperature at each session start.

---

## 10. Pilot run (step 7)

Run ID `c9-pilot-<YYYYMMDD>`. It is small but end-to-end: the full test-mode cycle, all model configurations, every condition type and every scorer. The output is kept in `results/raw/` but **never used as evidence**.

| Part | Content | Approx. time |
|---|---|---|
| Test mode | 9.0 → 9.1 → 9.2 → 9.3 at the start; 9.4 including `health.sh` at the end | 30 min |
| Accuracy smoke | M1 to M6, P1 to P3: NYC 0-shot 8 items (2 per fold), S-sim 8 (4 seen + 4 unseen), S-fixed 4, curated 5, Spider 10, BIRD 10 (M5 also native prompt 5 + 5); about 45 items per configuration | 9 configs x about 3 min = 30 min |
| Determinism | M1: 20 NYC items run twice in a row → byte-identical `raw` text expected | 2 min |
| LoRA smoke | F1 data, 20 optimiser steps (160 examples) under `power_log.py`. Copy adapter → convert on PC → `ollama create` with `ADAPTER` → 10 training-pool items and 10 eval items | 25 min + PC conversion |
| LoadGen smoke | M1 0-shot, `min_query_count 32`, `min_duration 60 s`; then M4 the same | 5 min |
| llama-bench | P1 (F16), one repetition | 3 min |
| Scoring on PC | all scorers on the pilot outputs; oracle runs: NYC gold → 100% strict; Spider gold → EX 1.000 and TS 1.000; BIRD gold → EX 100%; Arcwise gold re-score path runs | 20 min |

**The pilot must show all of the following before the full run starts:**
1. Test mode enters, verifies idle, and restores fully (`health.sh` ok, restore diff empty apart from transient units).
2. Every model loads at `100% GPU`. Each GGUF's SHA-256 matches `models.lock.json`, and the rendered prompt (saved) matches section 5 for every model's chat template.
3. Each model's extraction gives SQL for ≥ 95% of items. Cap hits are ≤ 5% per model; otherwise the prompt is reviewed before the full run, and any change is documented. Prem-1B-SQL prompt decision (section 5.4) taken.
4. No prompt exceeds `num_ctx − num_predict`.
5. The determinism check gives 20/20 identical outputs. If it does not, the identical rate is recorded and the team decides whether LoadGen-to-accuracy mapping for J/correct is acceptable.
6. LoRA: training runs with swap delta < 500 MB, and the s/step and peak RAM are recorded. The adapter changes outputs, with training-pool accuracy clearly above base. The Ollama `ADAPTER` path works, or the fallback is chosen.
7. LoadGen: the result `model` equals the requested tag. The summary says VALID (for the reduced settings). Timestamped energy integration agrees with mean-power x duration within 2%.
8. Power log: VDD_IN samples at ≥ 9 Hz with monotonic timestamps. The GPU-frequency and alarm sampler works.
9. All oracle checks pass, TS runs on the PC (or a fallback location is chosen, see risks), and the Arcwise re-score produces a number.
10. Per-query time per model is recorded and the section 11 estimates are updated. If the full plan grows by more than 25%, the team reconsiders the optional blocks.

---

## 11. Run order and time estimates

Per-query means come from the exploratory accuracy logs (`results/raw/exploratory-2026-10-13/taxi_sql/runs/acc/*.jsonl`, field `total_s`):

| Model | NYC 0-shot (s) | NYC 3-shot (s) | Spider (s) | BIRD (s) |
|---|---|---|---|---|
| qc1.5 | 2.50 | 4.28 (inflated by 73 runaways; 2.6 assumed) | 1.73 | 2.51 |
| qc3 | 3.68 | 3.64 | 2.34 | 3.74 |
| llama3.2-3b | 3.62 | 4.21 | 2.41 | 3.82 |
| qwen3-4b | 5.06 | 4.51 | 2.85 | 4.57 |

Assumptions:
- XiYan-3B ≈ qc3 (same architecture). Prem-1B ≈ 0.95 x qc1.5. Our Q4 ≈ 1.05 x M1 (a larger file).
- Q8_0 ≈ 1.2 x and F16 ≈ 1.85 x M1. This is a decode-bandwidth estimate (llama-bench M1 tg128 = 39.1 tok/s), and the pilot replaces it.
- LoRA runtime adapter ≈ 1.2 x M1.
- Training: 1.69 s per example per epoch (exploratory: 174 steps x 8 examples in 2,346 s).
- Curated C = 70.
- Per-model overhead (copy check, load, 5 warm-ups, 120 s loaded-idle, cool-down up to 20 min) ≈ 0.4 h.

| Order | Block | Content | Est. Jetson time |
|---|---|---|---|
| prep | B0 (PC, plus Jetson outside test mode) | Rebuild DB and questions (identity check), folds, curated set, conversions and hashes, test-suite DB download, scorer env, code fixes | about 1 to 2 working days of PC and team time (curated review included); 0 h Jetson test mode |
| 0 | **Pilot** | Section 10 | **about 2.5 h** |
| 1 | B1 session start (x3 sessions) | 9.0 to 9.3, idle baselines | 3 x 0.5 = 1.5 h |
| 2 | **B3 core accuracy**, M6 → M1 → M2 → M5 → M3 → M4 (small to large) | NYC 0-shot 519, S-sim 968, Spider 1,034, BIRD 500 | 1.83 + 1.91 + 2.70 + 2.70 + 2.88 + 3.40 = 15.4 h, + 6 x 0.4 = **17.8 h** |
| 2 | B4 S-fixed (run inside B3 while each model is loaded) | 519 items x 6 models | **3.0 h** |
| 2 | B5 XiYan native prompt (inside B3, M5 slot) | NYC 0-shot 519 + BIRD 500 | **1.05 h** |
| 3 | **B7a LoRA training** F1 to F4 | about 555 examples x 1 epoch each, idle baseline and cool-down per fold | 4 x (16 + 5 + 15 min) = **2.4 h** |
| 4 | **B6 precision ladder** P3 → P2 → P1 (adapters are converted on the PC meanwhile) | 4 sets each | 2.0 + 2.3 + 3.5 + 1.2 = **9.0 h** |
| 5 | **B7b LoRA evaluation** L1 to L4 | 4 x (450 fold items + 70 curated) x 3.0 s, + overhead | 1.73 + 1.6 = **3.3 h** |
| 6 | **B8 performance** | llama-bench 9 GGUFs (0.5 h). LoadGen 0-shot, 9 configs at 1,024 queries: M1 43, M2 63, M3 62, M4 86, M5 63, M6 41, P2 51, P1 79, L1 51 min = 9.0 h. 2 extra repeats each for M1 and M4 (4.3 h). Cool-down and idle 13 x 15 min (3.25 h) | **17.0 h** |
| 7 | (opt) B8b LoadGen 3-shot S-sim | 6 Q4 models (358 min) + cool-down | **7.5 h** |
| 8 | (opt) B8c clocks-off sensitivity (last, since it changes clocks) | M1 and M4 LoadGen 0-shot + new idle baseline | **2.7 h** |
| 9 | (opt) L1 on Spider | 1,034 x 3.0 s | **0.9 h** |
| 10 | (opt, D1) OmniSQL-7B | 4 sets at about 9 s/query + LoadGen | **about 9.5 h** |

**Totals (Jetson time in test mode):**

| Scope | Hours |
|---|---|
| Pilot | about 2.5 h |
| **Core** (B1, B3, B4, B5, B7a, B6, B7b, B8) | 1.5 + 17.8 + 3.0 + 1.05 + 2.4 + 9.0 + 3.3 + 17.0 ≈ **55 h** |
| Core + optional B8b, B8c, L1-Spider | ≈ **66 h** |
| With 15% contingency (re-runs of flagged LoadGen runs, resumptions) | core ≈ **63 h**, full ≈ **76 h** |
| OmniSQL-7B, if approved | +9.5 h |

**Sessions:** three sessions of about 20 to 26 h (session 1: B3 to B5; session 2: B7a, B6, B7b; session 3: B8 and options). The agent is restored between sessions. Wall-clock time is about 4 to 5 days, including PC-side conversion and scoring between sessions.

**Monitoring:** each block runs as `nohup bash block_<id>.sh > block.log 2>&1 &` in `~/c9/<run-id>/`. The PC polls every 30 to 60 min over ssh: progress lines, `tegrastats` tail, swap and tj, and the flags file.

**Run IDs:** `c9-pilot-<date>`; then `c9-main-<date>-s<session>-<block>` (for example `c9-main-20261020-s1-B3`). A LoadGen run that is interrupted or flagged is kept and redone under a new id. Accuracy runs may resume in the same folder (outputs are deterministic and skip-by-id); the resumption is logged in `run_meta.json`.

**After each session:** `python tools/fingerprint.py` in the project folder.

---

## 12. What is reused from the exploratory pass, and why

| Item | Reused? | Reason |
|---|---|---|
| Model outputs, scores, LoadGen and power results | **No, none as evidence** | A resident LLM service was running. The 3-shot prompt was broken (runaways). LoadGen ran the wrong model for 3 of 4. There was no TS. The KV cache was q8_0 and repeat penalty was not controlled. The splits and folds change. |
| Exploratory per-query times | Yes, for planning only | They are the best available estimates; the pilot replaces them |
| Exploratory zero-shot Spider/BIRD EX | Sanity check only | The new zero-shot EX should be within a few points; a large gap triggers a check of prompt and extraction. Not reported as a result. |
| Code: `build_db.py`, `templates.py`, `gen_questions.py`, `eval_exec.py` | Yes, unchanged | Seeded and deterministic. Byte-identity of the regenerated questions proves it (section 3.1). |
| Code: `run_ollama.py`, `loadgen_bench.py`, `finetune.py`, `prep_public.py`, `jetson/*.sh` | Yes, revised | Fixes: chat prompt and examples as turns; explicit sampling options; extraction rule; per-query ns timestamps; prompt-length assertion; LoadGen model assertion and timestamped energy; fold-aware fine-tuning with no merge and adapter-only output; `LD_LIBRARY_PATH` for cuDSS; TS scoring; no `$SUDO_PW` |
| Benchmark downloads in `~/bench` (Spider zip, Mini-Dev zip, evaluator clones, llama.cpp build, cuDSS, ftenv) | Yes, after hash and commit checks | Hashes and commits are recorded in section 3. Re-downloading adds nothing. |
| Exploratory LoRA adapter (`runs/ft-qwc15/adapter`) | No | It used the old prompt format, 2 epochs and the old split |
| `~/taxi_sql` on the Jetson | Untouched | Its PC copy is fingerprinted. Deletion needs team sign-off (D15). |

---

## 13. Files the doer will create in step 5/6 (so someone else can run this)

```
experiment/
  PLAN.md                      this file
  prior-code/                  unchanged
  code/   build_db.py templates.py gen_questions.py eval_exec.py (copied unchanged)
          make_folds.py prompts.py retrieval.py run_ollama.py loadgen_bench.py finetune.py
          prep_public.py score_spider.sh score_bird.sh score_arcwise.py nn_baseline.py stats.py
          power_log.py gpu_sampler.sh cooldown.sh testmode.sh env_record.sh
          block_B3.sh ... block_B8.sh  (exactly as run)
  config/ folds.json models.lock.json data.lock.json conditions.json generation.json loadgen_user.conf
  prompts/ rendered_examples.txt xiyan_nl2sqlite_cn.txt
  curated/ curated.jsonl DRAFTING.md REVIEW.md
  envs/   eval-requirements.txt convert-requirements.txt  (pinned, with lock files)
results/raw/<run-id>/   env/  acc/  perf/  train/  power/  logs/  run_meta.json   (as produced)
```

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The agent stack restarts mid-session (supervisor, @reboot) | No reboots during sessions. Port and process check every poll; any reappearance flags the block and pauses runs. |
| Ollama's `ADAPTER` with a Q4 base misbehaves or slows decoding sharply | Pilot check (10, item 6). Fallback is a merged-then-quantised GGUF on the PC (section 7). Runtime adapter overhead is measured in LoadGen L1. |
| LoRA training swaps or runs out of memory (exploratory: 7.5 GB RAM and 3.3 GB swap with the agent resident) | Agent and gdm stopped and swap flushed first. Batch 1 with answer-only logits (kept). No in-process merge. If swap delta > 500 MB, set max length to 768 (the mean example is 575 tokens) and document. |
| The test-suite evaluator or BIRD evaluator fails on Windows (multiprocessing, func_timeout) | Pilot check. Fallbacks in order: WSL on the PC; then the Jetson CPU **after** all measurement blocks, outside test mode, with the location recorded. |
| The TLC parquet changed, so the regenerated questions differ | Byte-identity check (3.1). If it fails, stop and report. The team decides whether to use the exploratory files as the frozen source; their hashes are in `MANIFEST.sha256`. |
| A Prem-1B-SQL or XiYan chat template is wrong in GGUF metadata | Rendered-template check in the pilot. Fix by setting an explicit `TEMPLATE` in the Modelfile, documented. |
| Non-determinism (prefix-cache reuse) breaks the LoadGen-to-accuracy mapping | Pilot determinism check. If not identical, compute J/correct from the accuracy runs' own per-query energy (which needs no mapping) and use LoadGen only for latency. |
| Thermal drift across days | Per-session idle baseline and T_idle-based cool-down. tj and GPU clock logged. Flagged LoadGen runs repeated. |
| Runaway generations inflate time and energy | New prompt format, cap 384, cap-hit rate reported; pilot gate ≤ 5% |
| Time overrun | Optional blocks are separable. Minimum LoRA is F1 and F2. S-fixed can be dropped (RQ3 still answered by S-sim). Decision D17. |
| Curated set falls below 60 after checks | Draft 80. Draft more before the freeze, never after. |
| Licence ambiguity (Qwen2.5-Coder-3B and XiYan derivative, Prem MIT vs Apache) | Recorded in the models table. The decision table marks commercial-use status per model. The team confirms (D2). |
| The corrected Mini-Dev changes the ranking | That is the point of the check. Report both, with τ. |
| Another user sends requests to Ollama | Bound to 127.0.0.1 in test mode. Per-request client logging. Ollama journal checked for foreign requests after each block. |

---

## 15. Decisions that need the team's sign-off

| # | Decision | Proposal |
|---|---|---|
| D1 | Model set | Add **XiYanSQL-QwenCoder-3B-2504** (primary specialised) and **Prem-1B-SQL** (secondary). Exclude FINER-SQL (no licence), SLM-SQL (CC-BY-NC) and CodeS (format and base model). **OmniSQL-7B optional** (+9.5 h). |
| D2 | Licence handling | Qwen2.5-Coder-3B is under the Qwen Research License (non-commercial). XiYan-3B's Apache tag is on a derivative of it. Prem-1B-SQL is tagged Apache but its text says MIT. Accept these models for research, and mark commercial-use status in the decision table? |
| D3 | NYC design | 4 rotating folds (F1 = original 6 construct-novel templates; F2 to F4 = 20 templates); eval set 449 items with m_t = min(20, ⌊n/2⌋); within-item seen/unseen pairing; **S-sim retrieval as the primary 3-shot**, S-fixed secondary |
| D4 | Hypothesis operationalisation and thresholds | H1 and H2 tested at 3-shot and LoRA (zero-shot has no seen/unseen); H1 ≥ 10 pp; H4 ≤ 3 pp upper bound; H6 seen ≥ 15 pp and unseen ≤ one third of seen; H5 "mid-sized" = 3B class |
| D5 | New prompt format (system schema, examples as chat turns, explicit greedy settings incl. `repeat_penalty 1.0`) | Approve. As a consequence, none of the exploratory outputs are reusable. |
| D6 | Power and clock state | MAXN_SUPER and **jetson_clocks ON** for all main runs, recorded per run, with the optional clocks-off sensitivity on 2 models. After the campaign: leave locked (state found) or reboot to restore DVFS? |
| D7 | Ollama KV cache during tests | **f16** (removes a confound for RQ5), vs the deployed q8_0. Also bind Ollama to loopback in test mode. |
| D8 | Test-mode scope | Stop the jetson-agent stack (LFM2.5 llama-server, opencode, proxy, workflow), gdm, maintenance timers, cron (pauses `ec2_healthcheck`), fwupd and packagekit; hold snap refreshes. Leave ssh-tunnel and docker running. Restore after each session. **Agent downtime: three sessions of about 20 to 26 h.** |
| D9 | Spider and BIRD at zero-shot only | Approve (justification in 4.1) |
| D10 | LoadGen scope | 1,024 queries and ≥600 s per configuration (meets the 2020 and current minimums); 1 run per configuration plus 3 repeats for M1 and M4. Is 3-shot LoadGen (+7.5 h) wanted? |
| D11 | LoRA | 1 epoch (exploratory loss collapsed by step 30), all 4 folds (minimum F1 and F2), runtime adapter on the identical M1 base, conversion on the PC; optional merged variant and L1-on-Spider |
| D12 | Curated (AI-drafted) set | Procedure in 3.3; **who drafts, and which team member reviews and signs `REVIEW.md`**; target 80 drafts and ≥60 accepted; 15 seed items kept as a flagged subset |
| D13 | Scoring location | All scoring on the PC (WSL as fallback, then Jetson after measurements) |
| D14 | Decision-table targets | Latency and energy budgets for RQ6, for example p90 ≤ 5 s and ≤ X J per correct answer. The team should set these values before results are seen. |
| D15 | Board clean-up | Delete the legacy `~/taxi_sql` and the Ollama models we create, after the campaign? (`~/bench` kept as fixtures.) |
| D16 | Corrected BIRD | Use Arcwise-Plat-SQL (CC BY-SA 4.0) as the sensitivity re-score; do not use Mini-Dev V2 or Arcwise-Plat full |
| D17 | Budget | Core about 55 h (63 h with contingency) vs full about 66 h (76 h). Choose the optional blocks. |

---

## Sign-off record (step 5)

Approved by the team on 2026-10-16 (tracker, step s05), accepting Claude's recommendation for every decision:

| # | Decision as approved |
|---|---|
| D1 | Add XiYanSQL-QwenCoder-3B-2504 and Prem-1B-SQL. OmniSQL-7B not included (outside the sub-4B scope). |
| D2 | All listed models kept for research use; the paper states each licence, including the non-commercial Qwen Research License on Qwen2.5-Coder-3B and the licence questions on XiYan and Prem. |
| D3 | Four rotating held-out folds, 449-item eval set, retrieved examples as the primary three-shot condition (fixed examples secondary). |
| D4 | Hypothesis thresholds as proposed (H1 >= 10 pp, H4 <= 3 pp, H6 15 pp / one third). |
| D5 | New chat-format prompt (schema in system message, examples as turns). No exploratory output reused as evidence. |
| D6 | jetson_clocks on and MAXN_SUPER for all main runs, recorded per run. After the campaign, reboot to restore normal clock scaling. |
| D7 | Ollama with f16 KV cache and loopback-only binding during tests; the team's settings restored afterwards. |
| D8 | Test mode approved: the jetson-agent stack, timers and cron paused in up to three windows of 20-26 h, restored and health-checked between windows. |
| D9 | Spider and BIRD at zero-shot only. |
| D10 | Core LoadGen scope plus three-shot LoadGen (+7.5 h). |
| D11 | LoRA: 1 epoch, four folds (minimum two), adapter applied at runtime without merging. |
| D12 | Curated set: about 80 items drafted by the agents, each with two independently written gold queries that must agree; a team member spot-checks 20 items before the set is frozen (presented at the pilot sign-off). |
| D13 | Scoring on the PC. |
| D14 | Decision-table targets, fixed before results: interactive = p90 latency <= 3 s; usable = p90 <= 10 s; accuracy tiers 50 %, 70 %, 90 % EX; energy per correct answer reported without a threshold. |
| D15 | Nothing deleted from the Jetson until the paper is published. |
| D16 | Corrected Mini-Dev (Jin et al., 498 items, CC BY-SA 4.0) as the BIRD sensitivity check. |
| D17 | Budget: core plan plus three-shot LoadGen, about 70 h of Jetson time. |
