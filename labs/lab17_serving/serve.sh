#!/usr/bin/env bash
# Lab 17, step 2: start one engine at a time on port 8000 with comparable settings.
#
#   bash serve.sh vllm      PagedAttention + continuous batching + prefix caching
#   bash serve.sh sglang    RadixAttention prefix cache + continuous batching
#   bash serve.sh llamacpp  llama.cpp server, GGUF, parallel slots (continuous batching)
#   bash serve.sh spec      vLLM with a small draft model for a larger target
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
MAXLEN=${MAXLEN:-4096}

case "${1:-vllm}" in
  vllm)
    vllm serve "$MODEL" --port 8000 --max-model-len "$MAXLEN" \
      --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 256 ;;
  sglang)
    python -m sglang.launch_server --model-path "$MODEL" --port 8000 \
      --context-length "$MAXLEN" --mem-fraction-static 0.85 ;;
  llamacpp)
    # -np: parallel slots; -c is TOTAL context shared by slots (so 32 x 4096 here)
    "${LLAMA:-$HOME/llama.cpp}/build/bin/llama-server" -m "${GGUF:-gguf/model-Q4_K_M.gguf}" \
      --port 8000 -np 32 -c $((32 * MAXLEN)) -ngl 99 --cont-batching ;;
  spec)
    # 0.5B drafter proposes 4 tokens per step for a 7B target (same tokenizer family).
    TARGET=${TARGET:-Qwen/Qwen2.5-7B-Instruct}
    DRAFT=${DRAFT:-Qwen/Qwen2.5-0.5B-Instruct}
    vllm serve "$TARGET" --port 8000 --max-model-len "$MAXLEN" \
      --speculative-config "{\"model\": \"$DRAFT\", \"num_speculative_tokens\": 4}" ;;
  *) echo "usage: serve.sh vllm|sglang|llamacpp|spec"; exit 1 ;;
esac
