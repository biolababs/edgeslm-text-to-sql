#!/bin/bash
# B7b LoRA evaluation L1..L4: runtime ADAPTER on the identical M1 base (qc15-q4-ollama). Adapters are converted on
# the Jetson at the end of each B7a fold (llama.cpp ecdd29e8 convert_lora_to_gguf.py) to ~/c9/models/<run>-lora-f<f>.gguf.
# Waits up to LORA_WAIT_S for each adapter GGUF; a missing adapter is flagged and its fold skipped. Sourced.
LORA_WAIT_S=${LORA_WAIT_S:-600}; [ "$DRY" = 1 ] && LORA_WAIT_S=60
i=0; echo "PROGRESS 0/4 B7b folds"
for f in 1 2 3 4; do
  g=$C/models/$(basename $RUN)-lora-f$f.gguf; w=0
  while [ ! -f $g.ready ] && [ $w -lt $LORA_WAIT_S ]; do
    [ $((w % 300)) -eq 0 ] && echo "PROGRESS $i/4 B7b waiting for adapter GGUF F$f (${w}s; PC conversion)"
    sleep 30; w=$((w + 30))
  done
  if [ ! -f $g.ready ]; then flag "B7b adapter GGUF F$f not delivered within ${LORA_WAIT_S}s - fold skipped"; continue; fi
  tag=qc15-lora-f$f-$(basename $RUN | tr '[:upper:]' '[:lower:]')
  printf 'FROM qc15-q4-ollama\nADAPTER %s\n' $g > $RUN/train/Modelfile.$tag
  sha256sum $g >> $RUN/train/adapter_gguf.sha256
  ollama create $tag -f $RUN/train/Modelfile.$tag 2>&1 | tail -n 1
  echo "F$f $tag" >> $C/models/lora_tags_$(basename $RUN).txt
  ev "BLOCK B7b model $tag (fold F$f)"
  run_model $tag lora${f}0 lora${f}sim cur0
  i=$((i + 1)); echo "PROGRESS $i/4 B7b folds"
done
ev "BLOCK B7b done"
