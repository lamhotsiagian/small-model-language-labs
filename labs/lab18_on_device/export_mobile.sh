#!/usr/bin/env bash
# Lab 18, step 2: package one fine-tuned model for three on-device runtimes.
#
#   gguf       llama.cpp / Ollama / LM Studio (laptops, Raspberry Pi, Android via termux or apps)
#   mlc        MLC-LLM: compiled TVM kernels for iOS (Metal), Android (OpenCL/Vulkan), WebGPU
#   executorch ExecuTorch .pte for Android/iOS with XNNPACK (CPU) or vendor NPU delegates
#
# Usage: MODEL=../lab11_sft/checkpoints/sft bash export_mobile.sh gguf|mlc|executorch
set -euo pipefail
MODEL=${MODEL:-../lab11_sft/checkpoints/sft}
OUT=${OUT:-mobile}
mkdir -p "$OUT"

case "${1:-gguf}" in
  gguf)
    LLAMA=${LLAMA:-$HOME/llama.cpp}
    python "$LLAMA/convert_hf_to_gguf.py" "$MODEL" --outtype f16 --outfile "$OUT/model-f16.gguf"
    "$LLAMA/build/bin/llama-quantize" "$OUT/model-f16.gguf" "$OUT/model-Q4_K_M.gguf" Q4_K_M
    ;;
  mlc)
    # 4-bit group quantisation (q4f16_1), conversation template from the tokenizer config
    mlc_llm convert_weight "$MODEL" --quantization q4f16_1 -o "$OUT/mlc"
    mlc_llm gen_config "$MODEL" --quantization q4f16_1 --conv-template chatml \
        --context-window-size 4096 --prefill-chunk-size 512 -o "$OUT/mlc"
    mlc_llm compile "$OUT/mlc/mlc-chat-config.json" --device android -o "$OUT/mlc/model-android.tar"
    # iOS:  --device iphone ;  browser: --device webgpu
    ;;
  executorch)
    # Llama-family export via ExecuTorch's LLM example tooling (see executorch/examples/models/llama).
    # 8da4w = 8-bit dynamic activations, 4-bit grouped weights; -X = XNNPACK delegate; kv cache + sdpa.
    python -m executorch.examples.models.llama.export_llama \
        --checkpoint "$MODEL/consolidated.00.pth" --params "$MODEL/params.json" \
        -kv --use_sdpa_with_kv_cache -X -qmode 8da4w --group_size 128 -d fp32 \
        --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
        --output_name "$OUT/model.pte"
    ;;
  *) echo "usage: export_mobile.sh gguf|mlc|executorch"; exit 1 ;;
esac
echo "[lab18] artifacts in $OUT/"
