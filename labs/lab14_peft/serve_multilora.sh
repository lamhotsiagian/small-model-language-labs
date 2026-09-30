#!/usr/bin/env bash
# Lab 14, step 5: serve three adapters from ONE base model with vLLM.
#
# One copy of the 3B base weights sits in GPU memory; each request names an
# adapter ("model" field), and vLLM batches requests for different adapters
# together using punica/S-LoRA style kernels (Sheng et al., 2024).
#
#   soc_lora_r8   the security-log classifier from this lab
#   soc_dora_r8   the DoRA variant
#   soc_qlora_r64 the QLoRA variant (adapter only; served on the bf16 base)
set -euo pipefail
BASE=${BASE:-meta-llama/Llama-3.2-3B-Instruct}

vllm serve "$BASE" \
  --enable-lora \
  --max-loras 3 \
  --max-lora-rank 64 \
  --lora-modules soc_lora_r8=checkpoints/lora_r8 \
                 soc_dora_r8=checkpoints/dora_r8 \
                 soc_qlora_r64=checkpoints/qlora_r64 \
  --max-model-len 2048 \
  --port 8000
# Note: DoRA adapters may need merging before serving if your vLLM version does
# not support them natively (python merge.py --adapter checkpoints/dora_r8 --merge-only).
