#!/bin/bash
# Session 3: B8 (llama-bench, LoadGen 0-shot + repeats) and B8b (LoadGen 3-shot). Usage: session_s3.sh <run-id> <S1 run-id> <S2 run-id>
set -u; RUN=$HOME/c9/runs/$1; S1RUN=$2; S2RUN=$3; mkdir -p $RUN/logs; source $HOME/c9/code/lib_blocks.sh
session_main 3 block_B8.sh
