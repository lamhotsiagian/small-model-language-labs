# Lab 2: Transformer From Scratch

**Chapter:** 2, Transformer Mathematics for Small Models
**Goal:** Implement a ~10M-parameter decoder in pure PyTorch (no Hugging Face
modelling code), train it on TinyStories, verify coherent generation, and
build a FLOPs and memory calculator whose predictions match measurement.

## Files

| File | Purpose |
|---|---|
| `attention_math.py` | Causal attention written as explicit tensor algebra; checks against the fused kernel; shows why we scale by 1/sqrt(d_k) |
| `train_tiny.py` | Trains `slmlab.model.SmallLM` (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings) with byte-level tokens and sequence packing |
| `calculator.py` | Analytic params / FLOPs / memory vs. `FlopCounterMode` measurement |
| `../../slmlab/model.py` | The model itself (shared by Labs 4, 5, 7, 9, 10) |

## Flow

1. **Derive attention.** `python attention_math.py`. Confirm the naive
   implementation matches `F.scaled_dot_product_attention` to ~1e-7 and read the
   variance/entropy table: unscaled logits saturate the softmax as d_k grows.
2. **Train.** `python train_tiny.py --preset smoke` (CPU, ~30 s) proves the
   pipeline; `python train_tiny.py --preset 10m` trains the 9.5M model on
   200K TinyStories documents (a single GPU, ~15-30 minutes).
   Install `datasets` to use real TinyStories; otherwise a synthetic corpus is used.
3. **Verify generation.** The script prefills "Once upon a time" and decodes
   160 tokens with the KV cache. Coherent grammar at 10M parameters is the
   pass criterion (Eldan & Li, 2023).
4. **Account.** `python calculator.py` prints analytic vs measured parameter
   count and linear-layer FLOPs (ratio must be 1.000) plus the analytic
   attention term and the backward/forward ratio (the 6N rule assumes 2.0).
5. **Scale the calculator.** Run it with Llama-3.2-1B dimensions and
   `--skip-measure` to see weights, optimizer state, KV cache per token, and
   the attention share of FLOPs at 8K context.

## Deliverable

Working training code, a generation sample, and the calculator report showing
analytic/measured ratios.

## Reference output (CPU, smoke preset, synthetic corpus)

```
params                       9,541,632       9,541,632   1.000
fwd_linear               4,882,759,680   4,882,759,680   1.000
train_linear            14,648,279,040  14,648,279,040   1.000
backward / forward = 2.00  (the 6N rule assumes 2.00)
```
