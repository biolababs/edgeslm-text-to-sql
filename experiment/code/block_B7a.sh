#!/bin/bash
# B7a on-box LoRA training, folds F1..F4, 1 epoch (D11), idle baseline before each run. Sourced.
i=0; echo "PROGRESS 0/4 B7a folds"
for f in 1 2 3 4; do
  if [ -f $RUN/train/f$f/adapter/adapter_model.safetensors ]; then ev "B7a fold F$f already trained, skipping"; i=$((i+1)); continue; fi
  baseline train_f$f 300
  ms=""; [ "$DRY" = 1 ] && ms="--max-steps 1"
  ev "BLOCK B7a lora_train F$f start swap=$(swap_used)MB"
  ( export LD_LIBRARY_PATH=$HOME/bench/cudss/lib:${LD_LIBRARY_PATH:-}; cd $CODE
    $HOME/bench/ftenv/bin/python finetune.py --base $HOME/bench/hf/qwc15 --pool $C/data/nyc_pool.jsonl --fold F$f \
      --schema $C/data/schema.sql --out $RUN/train/f$f $ms ) 2>&1 | tee $RUN/logs/lora_train_f$f.log
  ev "BLOCK B7a lora_train F$f end swap=$(swap_used)MB"
  sha256sum $RUN/train/f$f/adapter/adapter_model.safetensors >> $RUN/train/adapters.sha256
  # Deviation (step 9, coordinator): convert on the Jetson, not the PC, so the run does not depend on a live agent.
  # Same converter: llama.cpp ecdd29e8 convert_lora_to_gguf.py with its own gguf-py, base = pinned qwc15 safetensors.
  g=$C/models/$(basename $RUN)-lora-f$f.gguf
  ev "BLOCK B7a convert F$f start (llama.cpp $(git -C $HOME/bench/llama.cpp rev-parse --short HEAD))"
  ( cd $HOME/bench/llama.cpp && LD_LIBRARY_PATH=$HOME/bench/cudss/lib PYTHONPATH=$HOME/bench/llama.cpp/gguf-py       nice -n 10 $HOME/bench/ftenv/bin/python convert_lora_to_gguf.py $RUN/train/f$f/adapter --base $HOME/bench/hf/qwc15       --outtype f16 --outfile $g ) > $RUN/logs/lora_convert_f$f.log 2>&1
  rc=$?; ev "BLOCK B7a convert F$f end rc=$rc"
  if [ $rc -eq 0 ]; then sha256sum $g >> $RUN/train/adapter_gguf.sha256; touch $g.ready; else flag "LoRA conversion failed F$f"; fi
  swapflush; cooldown
  i=$((i + 1)); echo "PROGRESS $i/4 B7a folds"
done
ev "BLOCK B7a done"
