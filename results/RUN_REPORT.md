# Run report: C9 steps 6 and 7 (readiness and pilot), 2026-10-16

Full details: `experiment/PILOT_REPORT.md`. No evidence runs yet; step 9 has not started.

## Runs
| Run folder | What | Status |
|---|---|---|
| `results/raw/exploratory-2026-10-13/` | earlier exploratory pass (planning only) | unchanged |
| `results/raw/tmtest-20261016/` | step 6 trial of the test-mode procedure: enter, 180 s idle verification, 2-request smoke on M1, template check, 60 s baseline, exit with `health.sh` | pass; services restored |
| `results/raw/pilot-20261016/` | step 7 pilot (PLAN 10), parts A and B on the Jetson, plus `pc/` (LoRA conversion log and hash) | pass, with the notes in PILOT_REPORT section 4 |

Each folder holds `SHA256SUMS.jetson`, as produced on the Jetson and verified after the copy. `results/MANIFEST.sha256` covers 411 files and `fingerprint.py --check` passes.

## Contents of `pilot-20261016/`
- `env/`: pre and post snapshots, test-mode log, idle verification, baselines, environment records, Modelfiles, `ollama ps` output, restore diff.
- `acc/<model>/<cond>.jsonl`: raw outputs with ns timestamps and Ollama timings.
- `checks/`: template checks.
- `train/f1_smoke/`: adapter and training log.
- `perf/`: LoadGen logs, per-sample outputs and power, plus llama-bench output.
- `power/`: tegrastats ns logs and the GPU/alarm sampler.
- `logs/`: block logs, events, cool-downs.

## Commands
- Jetson: `nohup bash ~/c9/code/block_pilot_A.sh c9-pilot-20261016`, then `block_pilot_B.sh` (also the trial `block_tmtest.sh`). The code is deployed at `~/c9/code` and matches `experiment/code/MANIFEST.sha256` (sha `53ba13f7…`).
- PC: `score.py --run results/raw/pilot-20261016 --cfg experiment/config/score_paths.json --idle-w 6.512`, `score.py --oracle`, `pilot_checks.py`.

## Environment
Jetson Orin Nano Super, L4T R36.5.2, JetPack 6.2.3, MAXN_SUPER, jetson_clocks on, Ollama 0.30.11 (test mode: f16 KV, loopback only), llama.cpp `ecdd29e8` (b11602). PC: Python 3.12.15 envs, pinned in `experiment/envs/*.lock.txt`.

## Observations for the writer
Pilot numbers are not evidence. Problems found and the decisions they need are in PILOT_REPORT sections 4 and 5:
- Arcwise question changes;
- BIRD gold queries that exceed the 30 s timeout;
- the over-current alarm during training;
- swap creep;
- cross-context non-determinism;
- the Windows evaluator hangs, fixed outside the evaluators.

---

# Step 9: full experiment (in progress)

Approved at step 8 (sign-off record in `experiment/PILOT_REPORT.md`). Curated set frozen on 2026-10-17: `experiment/config/data.lock.json`, 92 items, `curated.jsonl` sha256 `a00ccd7b…`.

## Preparation
- **New requests** (`experiment/requests/MANIFEST.json`; all existing request files are unchanged):
  - `birdarc`: 129 corrected-BIRD items, i.e. the 127 with changed question or evidence plus 2 duplicated v1 ids, using Arcwise's own wording (sign-off item 1).
  - LoRA fold subsets `lora{1..4}0` and `lora{1..4}sim`.
  - Prompt-length gate re-run: 0 prompts over the limit.
- **Block scripts** (`experiment/code/`): `lib_blocks.sh`, `block_B3.sh` (B3 with B4, B5 and the extra corrected-BIRD run per model), `block_B6.sh`, `block_B7a.sh`, `block_B7b.sh`, `block_B8.sh` (B8, B8b, llama-bench), and session drivers `session_s1/s2/s3.sh`.
- **Approved changes applied:**
  - swap flush plus drop caches after every model, with the swap delta logged per model;
  - throttling flagged only on a clock drop or tj ≥ 95 °C (`flag_check.py`); crit_alarm samples are counted only;
  - the extra corrected-BIRD inference.
  - Energy per correct answer from the accuracy runs and the BIRD timeout handling are scoring-side changes.

## Deviations from PLAN (no measurement affected)
- **LoRA adapter conversion moved from the PC to the Jetson.** Each adapter is converted inside `block_B7a.sh` right after its training run, outside any measurement window. The converter is the same: llama.cpp `ecdd29e8` `convert_lora_to_gguf.py` with its own `gguf-py`, F16, base = the pinned `qwc15` safetensors. Reason: the full run must not depend on an agent being alive. In dry run 1 the PC-side delivery loop stalled and B7b waited 35 min.
- **The heartbeat is written by a detached PC daemon** (`hb_daemon.py`) that tails the session log. Its probe now uses no stdin, writes to a file instead of a pipe, and retries once, after an earlier version hung on a blocked ssh pipe.

## Bugs found by the dry runs and fixed before the full run
- **Hard-coded GPU sampler path in B8** (`gpu_s3.tsv`), caught by the monitor in dry run 1. `flag_check.py` crashed with FileNotFoundError, which was read as throttling, so every LoadGen run was falsely flagged and re-run (19 extra dry runs). Fix:
  - the path now comes from the sampler's own variable (`GPU_TSV`);
  - `flag_check.py` exits 2 with an `ERROR` line on a missing file or an empty window, which is never treated as throttled or as not throttled;
  - no session name is hard-coded anywhere.
- **Loop variable not local in `lg_run`.** It overwrote the outer repeat counter, so the M4 repeats were named `_repcur0`. Fixed by declaring the loop variables local.
- Dry run 1 is kept as `results/raw/dryrun-20261016/` (not evidence; 677 files, hash-verified). Dry run 2 (`c9-dryrun2-20261017`) re-tests B7a (on-Jetson conversion), B7b and B8.
- Dry run 2 passed: 0 flags, B7a conversion on the Jetson took about 8 s per fold, B7b found every adapter at once, run names were correct, and the exit `health.sh` returned 0. Kept as `results/raw/dryrun2-20261017/`. The dry runs' 1-step adapters are identical, because the first warm-up step has learning rate 0. That is expected and does not matter.

## Session log
- **Session 1** (`c9-main-20261017-s1`): started 2026-10-17 09:01 MDT. Blocks B3+B4+B5 plus the corrected-BIRD extra; expected about 20 h. Heartbeat comes from `hb_daemon.py`; the Jetson log is `/home/trafficedge01/c9/runs/c9-main-20261017-s1/logs/session.log`.

### Session 1 status, 2026-10-17 ~14:00 MDT
- B3: prem1b and qc15-q4-ollama done; qc3 running (model 3 of 6). One flag so far: swap delta 106 MB during prem1b, run kept and flagged per PLAN 9.5. Both finished models are being scored on the PC from a scratch copy; the official raw copy is made after the session.
- **Finding: host-RAM prompt cache drives memory and swap on long prompts.**
  - Ollama 0.30.11 starts `llama-server` without `--cache-ram`, so llama.cpp's default host prompt cache of **8,192 MiB** applies. Ollama exposes no setting for it. The team's own agent runs llama-server with `--cache-ram 0`.
  - Every distinct prompt's KV state is copied to host RAM. During prem1b's BIRD runs, used RAM rose from 5.0 to 7.2 GB and swap peaked at **936 MB**; the start/end delta of 106 MB understates this.
  - Prem uses full multi-head attention, about 196 KB of KV per token, and BIRD prompts reach 3,000 tokens. qc15 (GQA, about 28 KB/token) peaked at 2.9 GB RAM and 0 MB swap on the same prompts.
  - NYC and curated runs (520 to 1,150 tokens) stayed at 3 MB of swap or less for prem1b.
- **Impact:**
  - Accuracy: none.
  - Energy per correct answer: not affected. It is computed for NYC only (sign-off item 5), and NYC had no swap.
  - Per-query BIRD timing and energy: affected for prem1b (flagged), and probably for the larger-KV models still to come (llama3.2-3b, qwen3-4b, xiyan3b).
  - LoadGen (session 3): at risk. 1,024 queries over 541 prompts would fill the cache, so later queries would run under swap. Repeated prompts may also be restored from host RAM, which changes latency compared with a real deployment.
- **Action now:** none. Session 1 continues with the same server configuration for all six models, for consistency. Session 2 (P1-P3 and LoRA, all Qwen GQA models with small KV) is expected to be unaffected.
- **DECISION NEEDED (team) before session 3:**
  - Option A: disable the host prompt cache for LoadGen only, by setting `LLAMA_ARG_CACHE_RAM=0` in the test-mode Ollama drop-in. This matches the team's own agent and avoids swap; I would verify between sessions that Ollama passes the variable through.
  - Option B: keep the default and accept flagged LoadGen runs for the larger models.
  - My recommendation is A for session 3, reported as a stated serving setting. With A, the accuracy-run timings (default cache) and LoadGen (no cache) differ in that one setting, which the paper would state.
- Early scoring (scratch copy, PC): prem1b and qc15-q4-ollama scored with no null metrics (`score.py` exit 0). SQL extracted for 99.8-100% of items; cap hits at most 1.6% per condition (prem1b nyc0: 7/449). The full scoring runs again from the official raw copy after the session.
