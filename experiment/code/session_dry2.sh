#!/bin/bash
# Second dry run (after fixes): B7a with on-Jetson LoRA conversion, B7b, B8 (GPU_TSV fix). DRY=1. Usage: session_dry2.sh <run-id>
set -u; export DRY=1; RUN=$HOME/c9/runs/$1; S1RUN=c9-dryrun-20261016; S2RUN=$1; mkdir -p $RUN/logs; source $HOME/c9/code/lib_blocks.sh
session_main dry2 block_B7a.sh block_B7b.sh block_B8.sh
