#!/bin/bash
# B8 performance (PLAN 6.3, 11; D10): llama-bench on 9 GGUFs; LoadGen SingleStream 0-shot on 9 configs
# (1,024 queries, >= 600 s) + 2 extra repeats each of M1 and M4; then B8b LoadGen 3-shot S-sim on the 6 Q4 models.
# A run with a throttling flag (clock drop or tj >= 95 C) or a swap rise > 50 MB is kept and repeated once
# under a new id (_rerun) after a swap flush. Expected latency = mean accuracy-run latency of the same prompts.
# Needs S1RUN and S2RUN (run ids of sessions 1 and 2) for expected latencies and the L1 tag. Sourced.
MINQ=1024; MIND=600; [ "$DRY" = 1 ] && { MINQ=4; MIND=10; }
L1TAG=$(cat $C/models/lora_tags_$S2RUN.txt 2>/dev/null | awk '$1=="F1"{print $2}')
mean_lat() {  # tag conds-run files...
  local tag=$1 run=$2; shift 2
  python3 - "$C/runs/$run/acc/$tag" "$@" <<'PY'
import json, sys, statistics as s, os
d = sys.argv[1]; xs = []
for c in sys.argv[2:]:
    p = os.path.join(d, c + ".jsonl")
    if os.path.exists(p):
        xs += [json.loads(l)["wall_s"] for l in open(p)]
print(round(s.mean(xs), 3) if xs else 3.0)
PY
}
lg_run() {  # tag outname lat requests...
  local tag=$1 out=$2 lat=$3 attempt o; shift 3
  local req="" rq; for rq in "$@"; do req="$req $REQ/$rq.jsonl"; done
  for attempt in 1 2; do
    local o=$RUN/perf/$out; [ $attempt = 2 ] && o=$RUN/perf/${out}_rerun
    swapflush; cooldown
    agent_check; check_sha $tag
    local s0=$(swap_used) t0=$(date +%s%N)
    ev "LOADGEN $tag -> $(basename $o) start expected_latency=${lat}s swap=${s0}MB"
    ( cd $CODE; $HOME/bench/ftenv/bin/python loadgen_bench.py --model $tag --requests $req --gen $GEN --out $o \
        --min-queries $MINQ --min-duration-s $MIND --expected-latency-s $lat --idle-w $IDLEW ) 2>&1 | grep -v '^\s*$'
    local t1=$(date +%s%N) s1=$(swap_used)
    ollama ps > $o/ollama_ps.txt 2>&1; ollama stop $tag
    python3 $CODE/flag_check.py --gpu "$GPU_TSV" --t0 $t0 --t1 $t1 > $o/flag_check.json; local thr=$?
    if [ $thr -eq 2 ]; then flag "ERROR flag_check could not read the GPU sampler log ($GPU_TSV) for $(basename $o) - throttling unknown, not re-run"; thr=0; fi
    echo "{\"swap_start_mb\": $s0, \"swap_end_mb\": $s1, \"t0_ns\": $t0, \"t1_ns\": $t1}" > $o/swap.json
    ev "LOADGEN $tag $(basename $o) end throttle_rc=$thr swap_delta=$((s1 - s0))MB $(cat $o/flag_check.json)"
    if [ $thr -ne 0 ] || [ $((s1 - s0)) -gt 50 ]; then
      flag "LoadGen $(basename $o) flagged (throttle_rc=$thr swap_delta=$((s1 - s0))MB)"
      [ $attempt = 1 ] && continue
    fi
    break
  done
}

# ---- llama-bench, 9 GGUFs (no model loaded in Ollama)
for m in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop $m; done
mkdir -p $RUN/perf/llama-bench
i=0; echo "PROGRESS 0/9 B8 llama-bench"
for tag in qc15-q4-ollama qc3-q4-ollama llama32-3b-q4-ollama qwen3-4b-q4-ollama xiyan3b-q4-ours prem1b-q4-ours qc15-q4-ours qc15-q8-ours qc15-f16-ours; do
  f=$(ollama show --modelfile $tag | awk '/^FROM /{print $2; exit}')
  swapflush; cooldown
  args="-p 512 -n 128 -r 5"; [ "$DRY" = 1 ] && args="-p 32 -n 16 -r 1"
  ev "llama-bench $tag start ($f)"
  ( export LD_LIBRARY_PATH=$HOME/bench/cudss/lib:${LD_LIBRARY_PATH:-}
    $HOME/bench/llama.cpp/build/bin/llama-bench -m $f $args -ngl 99 -o json ) > $RUN/perf/llama-bench/$tag.json 2> $RUN/perf/llama-bench/$tag.stderr
  ev "llama-bench $tag end rc=$?"
  i=$((i + 1)); echo "PROGRESS $i/9 B8 llama-bench"
done

# ---- LoadGen 0-shot: 9 configs, then 2 extra repeats each for M1 and M4
CONF0="prem1b-q4-ours qc15-q4-ollama qc3-q4-ollama xiyan3b-q4-ours llama32-3b-q4-ollama qwen3-4b-q4-ollama qc15-q8-ours qc15-f16-ours"
i=0; N0=13; echo "PROGRESS 0/$N0 B8 LoadGen 0-shot"
for tag in $CONF0; do
  run=$S1RUN; case $tag in qc15-q8-ours|qc15-f16-ours) run=$S2RUN;; esac
  lg_run $tag loadgen0_$tag $(mean_lat $tag $run nyc0 cur0) nyc0 cur0
  i=$((i + 1)); echo "PROGRESS $i/$N0 B8 LoadGen 0-shot"
done
if [ -n "$L1TAG" ]; then
  lg_run $L1TAG loadgen0_qc15-lora-f1 $(mean_lat $L1TAG $S2RUN lora10 cur0) nyc0 cur0
else flag "L1 tag not found (lora_tags_$S2RUN.txt) - L1 LoadGen skipped"; fi
i=$((i + 1)); echo "PROGRESS $i/$N0 B8 LoadGen 0-shot"
for r in 2 3; do
  for tag in qc15-q4-ollama qwen3-4b-q4-ollama; do
    lg_run $tag loadgen0_${tag}_rep$r $(mean_lat $tag $S1RUN nyc0 cur0) nyc0 cur0
    i=$((i + 1)); echo "PROGRESS $i/$N0 B8 LoadGen 0-shot"
  done
done
ev "BLOCK B8 done"

# ---- B8b LoadGen 3-shot S-sim (D10)
baseline b8b_start 300
i=0; echo "PROGRESS 0/6 B8b LoadGen 3-shot"
for tag in prem1b-q4-ours qc15-q4-ollama qc3-q4-ollama xiyan3b-q4-ours llama32-3b-q4-ollama qwen3-4b-q4-ollama; do
  lg_run $tag loadgen3_$tag $(mean_lat $tag $S1RUN nycsim cursim) nycsim cursim
  i=$((i + 1)); echo "PROGRESS $i/6 B8b LoadGen 3-shot"
done
ev "BLOCK B8b done"
