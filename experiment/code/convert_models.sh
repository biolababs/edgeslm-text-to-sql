#!/bin/bash
# PC-side GGUF conversion (PLAN 2.2). Run from Git Bash on the PC.
#   convert_hf_to_gguf.py from llama.cpp ecdd29e8 (tag b11602), PC convert env (torch 2.11.0+cpu, transformers 4.57.6)
#   llama-quantize.exe from the official b11602 Windows CPU release (built from commit ecdd29e8), no imatrix
set -euo pipefail
C=/c/Users/biola/paper/models-cache/c9
LC=$C/src/llama.cpp                     # git checkout at ecdd29e8b2076bb85da9e9ca7ee3d84264375bd1
PY=$C/envs/convert/Scripts/python.exe
Q=$C/dl/llamacpp-b11602/bin/llama-quantize.exe
H=$C/hf; G=$C/gguf
mkdir -p $G
export PYTHONPATH="$(cygpath -w $LC/gguf-py)"
git -C $LC rev-parse HEAD
$Q --help 2>&1 | grep -m1 version || true

conv() {  # name hf_dir
  [ -f $G/$1-f16.gguf ] || $PY $LC/convert_hf_to_gguf.py "$(cygpath -w $H/$2)" --outtype f16 --outfile "$(cygpath -w $G/$1-f16.gguf)"
}
quant() { # name type suffix
  [ -f $G/$1-$3.gguf ] || $Q "$(cygpath -w $G/$1-f16.gguf)" "$(cygpath -w $G/$1-$3.gguf)" $2
}
conv xiyan3b xiyan3b; quant xiyan3b Q4_K_M q4_k_m
conv prem1b prem1b;   quant prem1b Q4_K_M q4_k_m
conv qc15 qc15;       quant qc15 Q8_0 q8_0; quant qc15 Q4_K_M q4_k_m
ls -l $G
sha256sum $G/*.gguf
echo CONVERT_DONE
