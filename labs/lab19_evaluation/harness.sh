#!/usr/bin/env bash
# Lab 19, step 2: standard benchmarks with lm-evaluation-harness, same settings for every model.
# Usage: bash harness.sh <model_path_or_id> [<more models> ...]
set -euo pipefail
TASKS=${TASKS:-mmlu_pro,hellaswag,arc_challenge,gsm8k,ifeval}
mkdir -p results/harness
for m in "$@"; do
  name=$(echo "$m" | tr '/' '_')
  lm_eval --model vllm \
    --model_args "pretrained=$m,gpu_memory_utilization=0.8,max_model_len=4096" \
    --tasks "$TASKS" --apply_chat_template --fewshot_as_multiturn \
    --batch_size auto --seed 1234 \
    --output_path "results/harness/$name" --log_samples
done
