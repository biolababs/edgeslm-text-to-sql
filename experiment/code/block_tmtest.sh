#!/bin/bash
# Step 6 readiness trial of the test-mode procedure (PLAN 9): precheck, enter, verify idle,
# a 2-request functional smoke on M1 (runner, template check), 60 s baseline, exit + health check.
#   nohup bash ~/c9/code/block_tmtest.sh c9-tmtest-20261016 > ~/c9/runs/c9-tmtest-20261016/logs/block.log 2>&1 &
set -u
RID=$1; C=$HOME/c9; CODE=$C/code; RUN=$C/runs/$RID; mkdir -p $RUN/{env,acc,logs,checks,power}
TOTAL=6; STEP=0; prog() { STEP=$((STEP + 1)); echo "PROGRESS $STEP/$TOTAL $*"; }
bash $CODE/testmode.sh precheck $RUN || { echo "BOARD BUSY - not entering"; exit 1; }; prog precheck
bash $CODE/testmode.sh enter $RUN; rc=$?; prog "enter rc=$rc"
if [ $rc -eq 0 ]; then
  bash $CODE/testmode.sh verify $RUN; prog "verify rc=$?"
  python3 $CODE/run_ollama.py --model qc15-q4-ollama --requests $C/requests/pilot_nyc0.jsonl --limit 2 --out $RUN/acc/qc15-q4-ollama/smoke.jsonl --gen $C/config/generation.json
  python3 $CODE/template_check.py --model qc15-q4-ollama --family qwen25coder15 --checks $C/requests/pilot_template_check.jsonl --gen $C/config/generation.json --out $RUN/checks/template_check_smoke.jsonl
  ollama ps > $RUN/env/ollama_ps_smoke.txt; ollama stop qc15-q4-ollama; prog smoke
  bash $CODE/testmode.sh baseline $RUN trial 60; prog baseline
fi
bash $CODE/testmode.sh exit $RUN; prog "exit rc=$?"
echo TMTEST_DONE
