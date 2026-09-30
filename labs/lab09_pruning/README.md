# Lab 9: Prune 3B to 1.5B

**Chapter:** 9, Pruning and Structured Compression
**Goal:** Compute layer and channel importance for a 3B model, prune to ~1.5B
with combined width and depth pruning, then recover with distillation on ~1B
tokens. Deliver a before/after benchmark table and a recovery curve.

## Files

| File | Purpose |
|---|---|
| `importance.py` | Forward hooks: block influence (1 - cos) per layer, FFN channel activation magnitude, per-head output norm; Wanda score helper; `--selftest` on a tiny random Llama |
| `prune.py` | Drops lowest-influence layers (first/last protected), slices FFN channels and KV groups, rewrites `config.json`; `--selftest` checks forward + generate |
| `recover.py` | Minitron-style logit KD from the unpruned parent on FineWeb-Edu, logs held-out CE and KL-to-teacher vs tokens |

## Pruning plan (Llama-3.2-3B, computed with `slmlab.budget`)

| Plan | Layers | d_ff | Heads / KV | Params |
|---|---|---|---|---|
| Original | 28 | 8192 | 24 / 8 | 3.21B |
| Depth only | 14 | 8192 | 24 / 8 | 1.80B |
| Width only (FFN x0.35) | 28 | 2880 | 24 / 8 | 1.84B |
| **Depth + width (default)** | **18** | **4096** | **24 / 8** | **1.53B** |
| Depth + width + KV x0.5 | 20 | 4096 | 12 / 4 | 1.40B |

The tied 128K embedding (394M) is untouched by layer/FFN pruning, which is why
depth-only pruning cannot reach 1.5B without removing more than half the layers.

## Flow

1. **Self-test.** `python importance.py --selftest && python prune.py --selftest`.
2. **Baseline.** `lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.2-3B --tasks mmlu,hellaswag,arc_challenge,winogrande`.
3. **Score.** `python importance.py --model meta-llama/Llama-3.2-3B --calib 256`.
   Plot block influence per layer: expect low scores in the upper-middle layers.
4. **Prune.** `python prune.py --model meta-llama/Llama-3.2-3B --importance results/importance.pt --drop-layers 10 --ffn-keep 0.5`.
5. **Measure the damage.** Run the same lm-eval tasks on the pruned model *before* recovery.
6. **Recover.** `python recover.py --teacher meta-llama/Llama-3.2-3B --student checkpoints/pruned_1p5b --tokens 1e9`.
7. **Re-evaluate** and plot `results/recovery_curve.json` (held-out CE and KL vs tokens).
8. **Ablate.** Repeat with depth-only and width-only plans at a matched size.

## Deliverable

Before/after table (original, pruned, recovered x {depth-only, width-only,
combined}) on MMLU, HellaSwag, ARC-C, WinoGrande, plus the recovery curve.
