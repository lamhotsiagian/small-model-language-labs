# Lab 16: Quantization Ladder

**Chapter:** 16, Quantization
**Goal:** Quantize one 1.5B model to FP16, INT8, AWQ 4-bit, GPTQ 4-bit, GGUF
Q4_K_M, Q3_K, and Q2_K. Measure perplexity, task accuracy, size, and
tokens/sec. Deliver a quality-vs-size Pareto chart and a recommended default.

## Files

| File | Purpose |
|---|---|
| `quant_math.py` | Symmetric / asymmetric, per-tensor / per-channel / per-group, NF4, and AWQ-style scaling on a weight with outlier channels (CPU) |
| `quantize_gpu.py` | bitsandbytes INT8; AWQ and GPTQ W4A16 via llm-compressor (servable by vLLM) |
| `gguf_ladder.sh` | llama.cpp: convert, imatrix, Q8_0 / Q5_K_M / Q4_K_M / Q3_K_M / Q2_K, perplexity, llama-bench |
| `pareto.py` | Pareto front and the "smallest within tolerance" recommendation |

## Flow

1. **Fundamentals.** `python quant_math.py`: see why per-tensor INT4 fails, why groups
   fix it, and what bits/weight the scales really cost.
2. **GPU formats.** `python quantize_gpu.py --method int8|awq|gptq --out quant/<m>`.
3. **GGUF ladder.** `bash gguf_ladder.sh` (CPU or Apple Silicon is fine; set `CALIB` to domain text).
4. **Measure.** For every artifact: WikiText-2 perplexity at 2K context, one task metric
   (`lm_eval --tasks arc_easy,gsm8k` for HF formats; the same prompts through
   `llama-server` for GGUF), size on disk, and decode tokens/s on your target hardware.
   Collect into `results/ladder.csv` (`name,size_mb,ppl,task_acc,tok_per_s`).
5. **Decide.** `python pareto.py --tolerance 1.0` -> `results/pareto.png` and a recommendation.
6. **Ablate the calibration set.** Re-run the imatrix / AWQ step with generic vs in-domain
   calibration text and compare the 3-bit results.

## Deliverable

Pareto chart, the ladder table, and a recommended default with a one-paragraph justification.

## Reference output (`python quant_math.py`, computed)

```
scheme                     bits/weight  weight err  output err
INT8 sym per-tensor              8.000      0.1101      0.1444
INT8 sym per-channel             8.016      0.0530      0.0696
INT4 sym per-tensor              4.000      0.6385      0.8226
INT4 sym group-128               4.125      0.2831      0.2280
INT4 sym group-32                4.500      0.1488      0.1295
NF4 group-64                     4.250      0.1505      0.1298
INT3 sym group-32                3.500      0.2257      0.2274
INT2 sym group-32                2.500      0.4395      0.5503
```
