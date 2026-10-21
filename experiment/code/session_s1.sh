#!/bin/bash
# Session 1: B3 (+B4, B5, corrected-BIRD extra).  setsid nohup bash ~/c9/code/session_s1.sh <run-id> > <run>/logs/session.log 2>&1 &
set -u; RUN=$HOME/c9/runs/$1; mkdir -p $RUN/logs; source $HOME/c9/code/lib_blocks.sh
session_main 1 block_B3.sh
