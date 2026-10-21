#!/bin/bash
# B3 core accuracy + B4 S-fixed + B5 XiYan native + corrected-BIRD extra (step 8 item 1), per model slot.
# Sourced by session_s1.sh (needs lib_blocks.sh). Order M6 -> M1 -> M2 -> M5 -> M3 -> M4 (small to large).
STD="nyc0 cur0 nycsim cursim nycfix curfix spider0 bird0 birdarc"
i=0
echo "PROGRESS 0/6 B3 models"
for tag in prem1b-q4-ours qc15-q4-ollama qc3-q4-ollama xiyan3b-q4-ours llama32-3b-q4-ollama qwen3-4b-q4-ollama; do
  ev "BLOCK B3 model $tag"
  if [ $tag = xiyan3b-q4-ours ]; then run_model $tag $STD nycnat curnat birdnat; else run_model $tag $STD; fi
  i=$((i + 1)); echo "PROGRESS $i/6 B3 models (done $tag)"
done
ev "BLOCK B3 done"
