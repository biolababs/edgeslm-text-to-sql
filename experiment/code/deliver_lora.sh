#!/bin/bash
# PC side of B7a -> B7b (PLAN 7): for each fold, wait until the Jetson has trained the adapter, copy it to the
# PC, convert with llama.cpp ecdd29e8 convert_lora_to_gguf.py (F16), copy the GGUF back as
# ~/c9/models/<run-id>-lora-f<f>.gguf and create the .ready marker that block_B7b.sh waits for.
#   bash deliver_lora.sh <run-id> [log]
set -u
RID=$1; LOG=${2:-/c/Users/biola/paper/models-cache/c9/logs/deliver_lora_$RID.log}
H="${JETSON_HOST:?set JETSON_HOST to user@host for the test bed}"; W=/c/Users/biola/paper/models-cache/c9/lora/$RID; mkdir -p $W
CODE=/c/Users/biola/paper/projects/c9-text-to-sql/experiment/code
for f in 1 2 3 4; do
  echo "PROGRESS $((f-1))/4 waiting for adapter F$f" >> $LOG
  until ssh -o BatchMode=yes -o ConnectTimeout=10 $H "grep -q \"F$f end\" ~/c9/runs/$RID/logs/events.tsv 2>/dev/null && test -f ~/c9/runs/$RID/train/f$f/adapter/adapter_model.safetensors"; do sleep 60; done
  sleep 5
  scp -q -r -o BatchMode=yes $H:c9/runs/$RID/train/f$f/adapter $W/f$f-adapter >> $LOG 2>&1
  bash $CODE/convert_lora.sh $W/f$f-adapter $W/lora-f$f.gguf >> $LOG 2>&1 || { echo "ERROR conversion failed F$f" >> $LOG; continue; }
  scp -q -o BatchMode=yes $W/lora-f$f.gguf $H:c9/models/$RID-lora-f$f.gguf >> $LOG 2>&1
  ssh -o BatchMode=yes $H "sha256sum ~/c9/models/$RID-lora-f$f.gguf && touch ~/c9/models/$RID-lora-f$f.gguf.ready" >> $LOG 2>&1
  echo "PROGRESS $f/4 delivered F$f" >> $LOG
done
echo "DELIVER_DONE" >> $LOG
