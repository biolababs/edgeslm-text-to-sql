#!/bin/bash
# PC-side LoRA adapter -> F16 LoRA GGUF (PLAN 7), llama.cpp ecdd29e8 convert_lora_to_gguf.py, CPU torch.
#   bash convert_lora.sh <adapter_dir> <out.gguf>
set -euo pipefail
C=/c/Users/biola/paper/models-cache/c9
export PYTHONPATH="$(cygpath -w $C/src/llama.cpp/gguf-py)"
$C/envs/convert/Scripts/python.exe $C/src/llama.cpp/convert_lora_to_gguf.py "$(cygpath -w $1)" \
  --base "$(cygpath -w $C/hf/qc15)" --outtype f16 --outfile "$(cygpath -w $2)"
sha256sum "$2"
