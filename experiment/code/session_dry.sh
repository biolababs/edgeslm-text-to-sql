#!/bin/bash
# 1-item dry run of every block in one short test-mode session (DRY=1). Usage: session_dry.sh <run-id>
set -u; export DRY=1; RUN=$HOME/c9/runs/$1; S1RUN=$1; S2RUN=$1; mkdir -p $RUN/logs; source $HOME/c9/code/lib_blocks.sh
session_main dry block_B3.sh block_B7a.sh block_B6.sh block_B7b.sh block_B8.sh
