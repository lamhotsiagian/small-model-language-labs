#!/usr/bin/env bash
# Lab 16, step 3: the GGUF k-quant ladder with llama.cpp.
#
# 1. convert the HF checkpoint to a 16-bit GGUF
# 2. compute an importance matrix (imatrix) on calibration text: per-weight
#    activation statistics that let low-bit k-quants spend precision where it matters
# 3. quantize to Q8_0, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K (imatrix used for <= 4 bits)
# 4. measure perplexity and speed for each file
#
# Requires a llama.cpp checkout built with `cmake -B build && cmake --build build -j`.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
LLAMA=${LLAMA:-$HOME/llama.cpp}
OUT=${OUT:-gguf}
CALIB=${CALIB:-data/calibration.txt}      # ~100-500 KB of in-domain text
EVAL=${EVAL:-data/wiki.test.raw}          # wikitext-2 test split
mkdir -p "$OUT" results

huggingface-cli download "$MODEL" --local-dir "$OUT/hf"
python "$LLAMA/convert_hf_to_gguf.py" "$OUT/hf" --outtype f16 --outfile "$OUT/model-f16.gguf"
"$LLAMA/build/bin/llama-imatrix" -m "$OUT/model-f16.gguf" -f "$CALIB" -o "$OUT/imatrix.dat" -c 512

for q in Q8_0 Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  extra=""
  [[ "$q" =~ ^Q[234] ]] && extra="--imatrix $OUT/imatrix.dat"
  "$LLAMA/build/bin/llama-quantize" $extra "$OUT/model-f16.gguf" "$OUT/model-$q.gguf" "$q"
done

echo "file,size_mb,ppl,pp512_tps,tg128_tps" > results/gguf_ladder.csv
for f in "$OUT"/model-*.gguf; do
  size=$(du -m "$f" | cut -f1)
  ppl=$("$LLAMA/build/bin/llama-perplexity" -m "$f" -f "$EVAL" -c 2048 --chunks 40 2>&1 \
        | grep -oE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$")
  bench=$("$LLAMA/build/bin/llama-bench" -m "$f" -p 512 -n 128 -o csv | tail -n 2)
  pp=$(echo "$bench" | head -n1 | awk -F, '{print $(NF-1)}')
  tg=$(echo "$bench" | tail -n1 | awk -F, '{print $(NF-1)}')
  echo "$(basename "$f"),$size,$ppl,$pp,$tg" | tee -a results/gguf_ladder.csv
done
