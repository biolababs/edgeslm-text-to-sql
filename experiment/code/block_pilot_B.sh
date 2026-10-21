#!/bin/bash
# Pilot part B (PLAN 10), still in test mode after part A. Needs ~/c9/models/lora-f1-smoke.gguf
# (adapter converted on the PC). LoRA ADAPTER path in Ollama, LoRA smoke eval (L1 vs M1),
# LoadGen smoke for M1 and M4, llama-bench P1, final baseline, exit test mode (restore + health).
#   nohup bash ~/c9/code/block_pilot_B.sh c9-pilot-20261016 > ~/c9/runs/c9-pilot-20261016/logs/block_B.log 2>&1 &
set -u
RID=$1
C=$HOME/c9; CODE=$C/code; REQ=$C/requests; CFG=$C/config; RUN=$C/runs/$RID
GEN=$CFG/generation.json
TOTAL=8; STEP=0
prog() { STEP=$((STEP + 1)); echo "PROGRESS $STEP/$TOTAL $*"; }
ev() { echo -e "$(date +%s%N)\t$*" >> $RUN/logs/events.tsv; echo "$(date -Is) $*"; }
TIDLE=$(python3 -c "import json; print(json.load(open('$RUN/env/baseline_session_start.json'))['tj_median_c'])")
IDLEW=$(python3 -c "import json; print(json.load(open('$RUN/env/baseline_session_start.json'))['vdd_in_median_w'])")
ev "block B start"

python3 $CODE/power_log.py --out $RUN/power/session_B.log --interval 100 & PL=$!
bash $CODE/gpu_sampler.sh $RUN/power/gpu_B.tsv & GS=$!

# ---- LoRA: register runtime adapter on the identical M1 base
printf 'FROM qc15-q4-ollama\nADAPTER %s\n' $C/models/lora-f1-smoke.gguf > $RUN/train/Modelfile.qc15-lora-f1-smoke
sha256sum $C/models/lora-f1-smoke.gguf > $RUN/train/lora-f1-smoke.gguf.sha256
ollama create qc15-lora-f1-smoke -f $RUN/train/Modelfile.qc15-lora-f1-smoke 2>&1 | tail -n 2
ollama show --modelfile qc15-lora-f1-smoke > $RUN/env/modelfile_qc15-lora-f1-smoke.txt 2>&1
ev "lora model created rc=$?"
python3 $CODE/make_lora_requests.py --train-log $RUN/train/f1_smoke/train_log.json --pool $C/data/nyc_pool.jsonl \
   --nyc0 $REQ/nyc0.jsonl --schema $C/data/schema.sql --fold F1 --out $RUN/train/pilot_lora.jsonl
prog lora_register
for tag in qc15-q4-ollama qc15-lora-f1-smoke; do
  bash $CODE/cooldown.sh $TIDLE $RUN/logs/cooldown.log
  ev "lora eval $tag start"
  python3 $CODE/run_ollama.py --model $tag --requests $RUN/train/pilot_lora.jsonl --out $RUN/acc/$tag/lora_smoke.jsonl --gen $GEN --warmup $REQ/warmup.jsonl
  ollama ps | tee $RUN/env/ollama_ps_lora_$tag.txt
  ollama stop $tag; ev "lora eval $tag end"
done
prog lora_eval

# ---- LoadGen smoke (session power log paused; loadgen_bench runs its own 100 ms log)
kill $PL; wait $PL 2>/dev/null
for tag in qc15-q4-ollama qwen3-4b-q4-ollama; do
  bash $CODE/cooldown.sh $TIDLE $RUN/logs/cooldown.log
  lat=$(python3 -c "import json,statistics as s; r=[json.loads(l) for l in open('$RUN/acc/$tag/nyc0.jsonl')]; print(round(s.mean(x['wall_s'] for x in r),3))")
  ev "loadgen $tag start expected_latency=$lat"
  ( cd $CODE; $HOME/bench/ftenv/bin/python loadgen_bench.py --model $tag --requests $REQ/nyc0.jsonl $REQ/cur0.jsonl --gen $GEN \
      --out $RUN/perf/loadgen_$tag --min-queries 32 --min-duration-s 60 --expected-latency-s $lat --idle-w $IDLEW )
  ev "loadgen $tag end rc=$?"
  ollama ps > $RUN/env/ollama_ps_loadgen_$tag.txt; ollama stop $tag
  prog loadgen_$tag
done
python3 $CODE/power_log.py --out $RUN/power/session_B2.log --interval 100 & PL=$!

# ---- llama-bench P1 (F16), one repetition
for m in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop $m; done
bash $CODE/cooldown.sh $TIDLE $RUN/logs/cooldown.log
ev "llama-bench start"
( export LD_LIBRARY_PATH=$HOME/bench/cudss/lib:${LD_LIBRARY_PATH:-}
  $HOME/bench/llama.cpp/build/bin/llama-bench -m $C/models/qc15-f16.gguf -p 512 -n 128 -r 1 -ngl 99 -o json ) > $RUN/perf/llama-bench_qc15-f16-ours.json 2> $RUN/logs/llama-bench.stderr
ev "llama-bench end rc=$?"
prog llama_bench

bash $CODE/cooldown.sh $TIDLE $RUN/logs/cooldown.log
kill $PL $GS 2>/dev/null
bash $CODE/testmode.sh baseline $RUN session_end 120
prog final_baseline
bash $CODE/env_record.sh $RUN qc15-q4-ollama qwen3-4b-q4-ollama qc15-lora-f1-smoke
echo "{\"run_id\": \"$RID\", \"block\": \"pilot-B\", \"end_utc\": \"$(date -u -Is)\"}" > $RUN/run_meta_B_end.json
prog env_record
bash $CODE/testmode.sh exit $RUN; rc=$?
ev "test mode exit rc=$rc"
prog exit
echo BLOCK_B_DONE rc=$rc
