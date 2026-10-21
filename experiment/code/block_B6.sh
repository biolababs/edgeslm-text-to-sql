#!/bin/bash
# B6 precision ladder P3 -> P2 -> P1 (no S-fixed, PLAN 4.1), plus corrected-BIRD extra. Sourced.
i=0; echo "PROGRESS 0/3 B6 models"
for tag in qc15-q4-ours qc15-q8-ours qc15-f16-ours; do
  ev "BLOCK B6 model $tag"
  run_model $tag nyc0 cur0 nycsim cursim spider0 bird0 birdarc
  i=$((i + 1)); echo "PROGRESS $i/3 B6 models (done $tag)"
done
ev "BLOCK B6 done"
