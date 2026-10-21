#!/bin/bash
# Pilot part A (PLAN 10): test mode in, idle verify, env record, idle baseline, LoRA training smoke,
# accuracy smoke for all 9 configurations (+ native prompt for M5, determinism for M1).
# Run on the Jetson:  nohup bash ~/c9/code/block_pilot_A.sh c9-pilot-20261016 > ~/c9/runs/c9-pilot-20261016/logs/block_A.log 2>&1 &
# Ends in test mode (part B follows after the PC-side LoRA conversion).
set -u
RID=$1
C=$HOME/c9; CODE=$C/code; REQ=$C/requests; CFG=$C/config; RUN=$C/runs/$RID
mkdir -p $RUN/{env,acc,perf,train,power,logs,checks}
GEN=$CFG/generation.json
TOTAL=15; STEP=0
prog() { STEP=$((STEP + 1)); echo "PROGRESS $STEP/$TOTAL $*"; }
ev() { echo -e "$(date +%s%N)\t$*" >> $RUN/logs/events.tsv; echo "$(date -Is) $*"; }
fail() { ev "FAIL $*"; echo "BLOCK_A_FAILED: $*"; exit 1; }

ev "block A start run=$RID"
echo "{\"run_id\": \"$RID\", \"block\": \"pilot\", \"start_utc\": \"$(date -u -Is)\", \"code_sha256_manifest\": \"$(sha256sum $CODE/MANIFEST.sha256 | cut -d' ' -f1)\"}" > $RUN/run_meta_A_start.json

bash $CODE/testmode.sh precheck $RUN || fail "precheck: board in use"
prog precheck
bash $CODE/testmode.sh enter $RUN || { bash $CODE/testmode.sh exit $RUN; fail "enter"; }
prog enter
bash $CODE/testmode.sh verify $RUN || { bash $CODE/testmode.sh exit $RUN; fail "idle verification"; }
prog verify
bash $CODE/env_record.sh $RUN qc15-q4-ollama qc3-q4-ollama llama32-3b-q4-ollama qwen3-4b-q4-ollama xiyan3b-q4-ours prem1b-q4-ours qc15-f16-ours qc15-q8-ours qc15-q4-ours
prog env_record
bash $CODE/testmode.sh baseline $RUN session_start 300
TIDLE=$(python3 -c "import json; print(json.load(open('$RUN/env/baseline_session_start.json'))['tj_median_c'])")
IDLEW=$(python3 -c "import json; print(json.load(open('$RUN/env/baseline_session_start.json'))['vdd_in_median_w'])")
ev "idle baseline tj=$TIDLE C VDD_IN=$IDLEW W"
prog baseline

# session samplers
python3 $CODE/power_log.py --out $RUN/power/session_A.log --interval 100 & PL=$!
bash $CODE/gpu_sampler.sh $RUN/power/gpu_A.tsv & GS=$!
trap 'kill $PL $GS 2>/dev/null' EXIT

# ---- LoRA training smoke (F1, 20 steps = 160 examples)
ev "lora_train start"
( export LD_LIBRARY_PATH=$HOME/bench/cudss/lib:${LD_LIBRARY_PATH:-}; cd $CODE
  $HOME/bench/ftenv/bin/python finetune.py --base $HOME/bench/hf/qwc15 --pool $C/data/nyc_pool.jsonl --fold F1 \
     --schema $C/data/schema.sql --out $RUN/train/f1_smoke --max-steps 20 ) > $RUN/logs/lora_train.log 2>&1
rc=$?; ev "lora_train end rc=$rc"
tail -n 3 $RUN/logs/lora_train.log
prog lora_train

expected_sha() { python3 -c "import json; m=json.load(open('$CFG/models.lock.json'))['models']['$1']; print(m['ollama_blob_sha256'])"; }
run_model() {  # tag family conds...
  tag=$1; fam=$2; shift 2
  bash $CODE/cooldown.sh $TIDLE $RUN/logs/cooldown.log; ev "cooldown done before $tag ($(tail -n1 $RUN/logs/cooldown.log))"
  blob=$(ollama show --modelfile $tag | awk '/^FROM /{print $2; exit}')
  got=$(sha256sum $blob | cut -d' ' -f1); exp=$(expected_sha $tag)
  [ "$got" = "$exp" ] && ev "sha ok $tag" || ev "FLAG sha mismatch $tag got=$got expected=$exp"
  ollama show --modelfile $tag > $RUN/env/modelfile_$tag.txt
  ev "load+warmup $tag"
  python3 $CODE/run_ollama.py --model $tag --requests $REQ/warmup.jsonl --out $RUN/acc/$tag/_warmup.jsonl --gen $GEN
  ollama ps | tee $RUN/env/ollama_ps_$tag.txt
  grep -q "100% GPU" $RUN/env/ollama_ps_$tag.txt && ev "100% GPU $tag" || ev "FLAG not 100% GPU $tag"
  ev "loaded_idle start $tag"; sleep 120; ev "loaded_idle end $tag"
  for c in "$@"; do
    ev "run $tag $c start"
    python3 $CODE/run_ollama.py --model $tag --requests $REQ/pilot_$c.jsonl --out $RUN/acc/$tag/$c.jsonl --gen $GEN
    ev "run $tag $c end"
  done
  if [ "$tag" = "qc15-q4-ollama" ]; then
    for k in 1 2; do ev "determinism $k start"
      python3 $CODE/run_ollama.py --model $tag --requests $REQ/pilot_determinism.jsonl --out $RUN/acc/$tag/determinism_run$k.jsonl --gen $GEN
      ev "determinism $k end"; done
  fi
  ev "template_check $tag start"
  python3 $CODE/template_check.py --model $tag --family $fam --checks $REQ/pilot_template_check.jsonl --gen $GEN --out $RUN/checks/template_check_$tag.jsonl
  ev "template_check $tag end"
  ollama stop $tag; sleep 3; ollama ps > $RUN/env/ollama_ps_after_$tag.txt
  ev "unloaded $tag"
}
STD="nyc0 nycsim nycfix cur0 spider0 bird0"
run_model prem1b-q4-ours prem1b $STD;                          prog prem1b
run_model qc15-q4-ollama qwen25coder15 $STD;                   prog qc15-q4-ollama
run_model qc3-q4-ollama qwen25coder3 $STD;                     prog qc3
run_model xiyan3b-q4-ours xiyan3b $STD nycnat birdnat curnat;  prog xiyan3b
run_model llama32-3b-q4-ollama llama32_3b $STD;                prog llama32
run_model qwen3-4b-q4-ollama qwen3_4b_2507 $STD;               prog qwen3
run_model qc15-q4-ours qwen25coder15 $STD;                     prog qc15-q4-ours
run_model qc15-q8-ours qwen25coder15 $STD;                     prog qc15-q8-ours
run_model qc15-f16-ours qwen25coder15 $STD;                    prog qc15-f16-ours
kill $PL $GS 2>/dev/null; trap - EXIT
echo "{\"run_id\": \"$RID\", \"block\": \"pilot-A\", \"end_utc\": \"$(date -u -Is)\", \"tidle_c\": $TIDLE, \"idle_w\": $IDLEW, \"nvpmodel\": \"$(nvpmodel -q | head -1)\", \"gpu_min_max_hz\": \"$(cat /sys/class/devfreq/17000000.gpu/min_freq)/$(cat /sys/class/devfreq/17000000.gpu/max_freq)\"}" > $RUN/run_meta_A_end.json
ev "block A done (still in test mode)"
echo BLOCK_A_DONE
