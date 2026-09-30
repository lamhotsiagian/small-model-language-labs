---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B
tags: [sft, small-language-model, instruction-tuned]
---

# <model-name>

## Summary
Instruction-tuned version of `<base_model>` (pinned revision `<sha>`), trained with
supervised fine-tuning on a curated 50K-example instruction set.

## Intended use
- In scope: <tasks, languages, deployment targets>
- Out of scope: <medical/legal advice, high-stakes decisions, languages not evaluated>

## Training data
| Source | Share | License | Notes |
|---|---|---|---|
| smoltalk | | | |
| tulu-3-sft-mixture | | | |
| magpie (self-synthesized) | | | |
Cleaning, dedup, and decontamination steps: see `build_dataset.py` stats.

## Training procedure
LR, epochs, effective batch, max length, packing, completion-only loss, hardware, GPU-hours.

## Chat template
The exact template shipped in `tokenizer_config.json`. Serving MUST use it.

## Evaluation
| Benchmark | This model | Official instruct | Base |
|---|---|---|---|
| IFEval (prompt-level strict) | | | |
| Pairwise vs official (judge, swap-consistent) | | -- | -- |
| Domain eval | | | |

## Limitations and risks
Known failure modes, hallucination behavior, refusal calibration, languages.
