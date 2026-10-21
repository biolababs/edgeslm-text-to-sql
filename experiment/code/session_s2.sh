#!/bin/bash
# Session 2: B7a LoRA training -> B6 precision ladder (adapters converted on the PC meanwhile) -> B7b LoRA eval.
set -u; RUN=$HOME/c9/runs/$1; mkdir -p $RUN/logs; source $HOME/c9/code/lib_blocks.sh
session_main 2 block_B7a.sh block_B6.sh block_B7b.sh
