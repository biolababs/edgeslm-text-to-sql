#!/bin/bash
# Shared functions for the main-run blocks (PLAN 11, sign-off records of steps 5 and 8).
# Sourced by session_*.sh. Requires RUN (run dir on the Jetson) and C (~/c9).
# DRY=1: 1-item dry run (short cool-downs/baselines, 1 request per condition, tiny LoadGen/LoRA).
C=${C:-$HOME/c9}; CODE=$C/code; REQ=$C/requests; CFG=$C/config; GEN=$CFG/generation.json
DRY=${DRY:-0}
mkdir -p $RUN/{env,acc,perf,train,power,logs,checks}
PL=; GS=; PLTAG=; GPU_TSV=
AGENT_PAT='llama-server|opencode|proxy/server.js|workflow/server.py|jetson-agent/scripts/run-server.sh'

ev()   { echo -e "$(date +%s%N)\t$*" >> $RUN/logs/events.tsv; echo "$(date -Is) $*"; }
flag() { echo -e "$(date +%s%N)\t$*" >> $RUN/logs/flags.tsv; ev "FLAG $*"; }
swap_used() { free -m | awk '/Swap/{print $3}'; }

# approved change (step 8, item 4): swap flush + drop caches in every between-model cool-down
swapflush() {
  local before=$(swap_used) avail=$(free -m | awk '/Mem/{print $7}')
  if [ $((avail - before)) -gt 1024 ]; then sudo swapoff -a && sudo swapon -a; fi
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
  ev "swapflush before=${before}MB after=$(swap_used)MB"
}

cooldown() {  # waits until tj <= T_idle + 2 C (min 180 s, max 1200 s)
  if [ "$DRY" = 1 ]; then sleep 5; ev "cooldown (dry) skipped"; return; fi
  bash $CODE/cooldown.sh $TIDLE $RUN/logs/cooldown.log
  local last=$(tail -n1 $RUN/logs/cooldown.log); ev "cooldown: $last"
  case "$last" in *FLAG*) flag "cooldown_max_reached";; esac
}

baseline() {  # name seconds; sets IDLEW/TIDLE when name=session_start. Pauses the session power log meanwhile.
  local s=$2; [ "$DRY" = 1 ] && s=15
  local resume=""; if [ -n "$PL" ]; then kill $PL 2>/dev/null; wait $PL 2>/dev/null; resume=1; fi
  bash $CODE/testmode.sh baseline $RUN $1 $s
  if [ -n "$resume" ]; then python3 $CODE/power_log.py --out $RUN/power/session_$PLTAG.log --interval 100 & PL=$!; fi
  if [ "$1" = session_start ]; then
    TIDLE=$(python3 -c "import json; print(json.load(open('$RUN/env/baseline_session_start.json'))['tj_median_c'])")
    IDLEW=$(python3 -c "import json; print(json.load(open('$RUN/env/baseline_session_start.json'))['vdd_in_median_w'])")
    ev "idle baseline tj=$TIDLE C VDD_IN=$IDLEW W"
  fi
}

load_baseline_vars() {
  TIDLE=$(python3 -c "import json; print(json.load(open('$RUN/env/baseline_session_start.json'))['tj_median_c'])")
  IDLEW=$(python3 -c "import json; print(json.load(open('$RUN/env/baseline_session_start.json'))['vdd_in_median_w'])")
}

start_samplers() {  # tag
  PLTAG=$1
  python3 $CODE/power_log.py --out $RUN/power/session_$1.log --interval 100 & PL=$!
  GPU_TSV=$RUN/power/gpu_$1.tsv
  bash $CODE/gpu_sampler.sh $GPU_TSV & GS=$!
  ev "samplers started $1 (power pid $PL, gpu pid $GS)"
}
stop_samplers() { [ -n "$PL$GS" ] && { kill $PL $GS 2>/dev/null; wait $PL 2>/dev/null; ev "samplers stopped"; }; PL=; GS=; }

agent_check() {
  if pgrep -f "$AGENT_PAT" > /dev/null; then flag "agent stack process reappeared"; pgrep -af "$AGENT_PAT" >> $RUN/logs/flags.tsv; fi
  if ss -tlnH | grep -qE '0\.0\.0\.0:11434|\*:11434'; then flag "ollama not loopback-only"; fi
}

check_sha() {  # tag
  local blob got exp
  blob=$(ollama show --modelfile $1 | awk '/^FROM /{print $2; exit}')
  got=$(sha256sum $blob | cut -d' ' -f1)
  exp=$(python3 -c "import json; m=json.load(open('$CFG/models.lock.json'))['models'].get('$1'); print(m['ollama_blob_sha256'] if m else 'none')")
  if [ "$exp" = none ]; then ev "sha $1 not in lock (adapter model): $got"
  elif [ "$got" = "$exp" ]; then ev "sha ok $1"; else flag "sha mismatch $1 got=$got expected=$exp"; fi
  ollama show --modelfile $1 > $RUN/env/modelfile_$1.txt
}

journal_snapshot() {  # name: Ollama journal since the block started (foreign request check)
  sudo journalctl -u ollama --since "@$(( $1 / 1000000000 ))" --no-pager > $RUN/env/ollama_journal_$2.txt 2>&1
  grep -E '\| *(POST|GET) ' $RUN/env/ollama_journal_$2.txt | grep -v '127.0.0.1' > $RUN/env/ollama_foreign_$2.txt
  [ -s $RUN/env/ollama_foreign_$2.txt ] && flag "non-loopback Ollama requests in $2"
}

# run_model TAG COND... : SHA check, warm-up, ollama ps (100% GPU), 120 s loaded idle, conditions, unload,
# swap flush (recorded per model), cool-down.
run_model() {
  local tag=$1 c; shift
  agent_check
  check_sha $tag
  local s0=$(swap_used); ev "model $tag start swap=${s0}MB"
  python3 $CODE/run_ollama.py --model $tag --requests $REQ/warmup.jsonl --out $RUN/acc/$tag/_warmup.jsonl --gen $GEN
  ollama ps | tee $RUN/env/ollama_ps_$tag.txt
  grep -q "100% GPU" $RUN/env/ollama_ps_$tag.txt && ev "100% GPU $tag" || flag "not 100% GPU $tag (block invalid per PLAN 2.2)"
  local li=120; [ "$DRY" = 1 ] && li=5
  ev "loaded_idle start $tag"; sleep $li; ev "loaded_idle end $tag"
  local lim=""; [ "$DRY" = 1 ] && lim="--limit 1"
  for c in "$@"; do
    ev "run $tag $c start"
    python3 $CODE/run_ollama.py --model $tag --requests $REQ/$c.jsonl --out $RUN/acc/$tag/$c.jsonl --gen $GEN $lim
    ev "run $tag $c end rc=$?"
  done
  ollama stop $tag; sleep 3
  local s1=$(swap_used); ev "model $tag end swap=${s1}MB delta=$((s1 - s0))MB"
  [ $((s1 - s0)) -gt 50 ] && flag "swap delta $((s1 - s0))MB during $tag (accuracy run kept, flagged)"
  swapflush
  cooldown
}

session_enter() {  # returns non-zero if the board is busy or entry/verification failed (test mode exited)
  bash $CODE/testmode.sh precheck $RUN || { ev "precheck: board busy - not entering"; return 1; }
  bash $CODE/testmode.sh enter $RUN || { bash $CODE/testmode.sh exit $RUN; ev "enter failed, restored"; return 1; }
  ENTERED=1
  bash $CODE/testmode.sh verify $RUN || { bash $CODE/testmode.sh exit $RUN; ENTERED=0; ev "idle verification failed, restored"; return 1; }
  bash $CODE/env_record.sh $RUN qc15-q4-ollama qc3-q4-ollama llama32-3b-q4-ollama qwen3-4b-q4-ollama xiyan3b-q4-ours prem1b-q4-ours qc15-f16-ours qc15-q8-ours qc15-q4-ours
  baseline session_start 300
}

session_exit() {
  bash $CODE/testmode.sh baseline $RUN session_end $([ "$DRY" = 1 ] && echo 15 || echo 120)
  bash $CODE/env_record.sh $RUN qc15-q4-ollama
  bash $CODE/testmode.sh exit $RUN; local rc=$?
  ENTERED=0
  ev "test mode exit rc=$rc (health.sh rc)"
  echo "SESSION_EXIT health_rc=$rc"
  return $rc
}

start_gpu_sampler() { GPU_TSV=$RUN/power/gpu_$1.tsv; bash $CODE/gpu_sampler.sh $GPU_TSV & GS=$!; ev "gpu sampler started $1 (pid $GS)"; }

session_main() {  # session-number block-files... : common driver (enter, blocks, exit; trap restores on failure)
  local sn=$1; shift
  ENTERED=0
  trap '[ "$ENTERED" = 1 ] && { stop_samplers; session_exit; }' EXIT
  trap 'ev "signal received - restoring"; exit 1' TERM INT HUP
  echo "{\"run_id\": \"$(basename $RUN)\", \"session\": \"$sn\", \"blocks\": \"$*\", \"dry\": $DRY, \"start_utc\": \"$(date -u -Is)\", \"code_manifest_sha256\": \"$(sha256sum $CODE/MANIFEST.sha256 | cut -d' ' -f1)\", \"S1RUN\": \"${S1RUN:-}\", \"S2RUN\": \"${S2RUN:-}\"}" > $RUN/run_meta_start.json
  ev "session $sn start blocks: $*"
  session_enter || { echo "SESSION_ABORTED"; trap - EXIT; return 1; }
  local T0=$(date +%s%N)
  for b in "$@"; do
    if [ "$b" = block_B8.sh ]; then start_gpu_sampler s$sn; else start_samplers s$sn-$(basename $b .sh); fi
    ev "BLOCKFILE $b start"; source $CODE/$b; ev "BLOCKFILE $b end"
    stop_samplers
  done
  journal_snapshot $T0 s$sn
  echo "{\"run_id\": \"$(basename $RUN)\", \"end_utc\": \"$(date -u -Is)\", \"tidle_c\": $TIDLE, \"idle_w\": $IDLEW, \"nvpmodel\": \"$(nvpmodel -q | head -1)\"}" > $RUN/run_meta_end.json
  session_exit; local rc=$?
  trap - EXIT
  echo "SESSION${sn}_DONE health_rc=$rc"
}
