# Lab 4: Architecture Ablations

**Chapter:** 4, SLM Architecture Design Choices
**Goal:** At a fixed ~125M parameters and a fixed token budget, train four
variants (wide-shallow, deep-thin, MHA vs GQA, tied vs untied) and compare
validation loss, inference throughput, and KV-cache size.

## Files

| File | Purpose |
|---|---|
| `ablate.py` | Builds parameter-matched variants (solves for layer count), trains each with identical data/seed/schedule, measures batched decode throughput and KV bytes/token |
| `plot_curves.py` | Validation loss vs tokens for all variants on one chart |

## Variants at the 125M preset (vocab 32K)

| Variant | d_model | Layers | Heads / KV | Tied |
|---|---|---|---|---|
| deep_thin | 576 | 31 | 9 / 3 | yes |
| wide_shallow | 1024 | 9 | 16 / 4 | yes |
| deep_thin_mha | 576 | 27 | 9 / 9 | yes |
| deep_thin_untied | 576 | 25 | 9 / 3 | no |
| deep_thin_qknorm (bonus) | 576 | 31 | 9 / 3 | yes, QK-norm |

Layer counts are solved so every variant lands on the same total parameter
budget: MHA and untied embeddings *cost layers*, which is the real trade-off.

## Flow

1. **Smoke test.** `python ablate.py --preset smoke` (CPU, a few minutes) proves the pipeline.
2. **Prepare data.** For the full preset, point `load_text_corpus` at FineWeb-Edu
   and swap in the 32K tokenizer from Lab 3. Budget ~2.6B tokens
   (20K steps x 32 x 4 x 1024).
3. **Train.** `python ablate.py --preset 125m`. Each variant sees the same
   tokens in the same order from the same initial seed.
4. **Plot.** `python plot_curves.py` writes `results/loss_curves.png`.
5. **Tabulate.** Fill the ablation table (val loss, decode tok/s, KV KB/token)
   from `results/ablation.json`.
6. **Interpret.** For each pair, state which dimension won and at what cost:
   depth usually wins on loss and loses on latency; GQA costs almost nothing
   in loss and cuts the KV cache 3x; tying frees ~18M parameters for layers.

## Deliverable

Ablation table plus loss curves, with a one-paragraph conclusion per question.
