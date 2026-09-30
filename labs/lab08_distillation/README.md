# Lab 8: Teacher-Student Distillation

**Chapter:** 8, Knowledge Distillation
**Goal:** Distil a 7-8B teacher into a 0.5B student on math reasoning (GSM8K)
using (a) SFT on teacher outputs, (b) logit KD, (c) on-policy GKD. Compare
accuracy and training cost across the three methods.

## Files

| File | Purpose |
|---|---|
| `losses.py` | Forward KL, reverse KL, generalised JSD (GKD), top-k offline KL, SFT CE; `python losses.py` runs the mode-seeking demo |
| `gsm8k.py` | Prompt format, `#### n` answer extraction, accuracy loop |
| `gen_teacher.py` | vLLM teacher sampling with rejection sampling (correct traces only); optional top-k logprob storage |
| `train_kd.py` | `--method sft | logit | gkd`, online or `--offline` top-k teacher |
| `evaluate.py` | GSM8K test accuracy for all checkpoints (vLLM, greedy) |

Default pair: teacher `Qwen/Qwen2.5-7B-Instruct`, student `Qwen/Qwen2.5-0.5B-Instruct`
(same tokenizer). Alternative: `meta-llama/Llama-3.1-8B-Instruct` -> `meta-llama/Llama-3.2-1B-Instruct`.

## Flow

1. **Understand the objective.** `python losses.py` fits a one-bump student to a
   two-mode teacher: forward KL spreads mass across both modes and the gap,
   reverse KL and GKD (beta=0.9) lock onto one mode.
2. **Generate teacher data.** `python gen_teacher.py --k 4 --out data/teacher.jsonl`
   (keep one correct trace per question). Record the coverage rate.
3. **Train three students** from the same base, same data, same steps:
   ```bash
   python train_kd.py --method sft   --data data/teacher.jsonl
   python train_kd.py --method logit --data data/teacher.jsonl
   python train_kd.py --method gkd   --data data/teacher.jsonl --beta 0.5 --lam 0.5
   ```
4. **Evaluate.** `python evaluate.py --models <base> checkpoints/student_* <teacher>`.
5. **Account for cost.** For each method record GPU-hours, peak memory, and
   teacher forward passes (logged by `train_kd.py`). Logit KD holds the teacher
   in memory; GKD also pays for student generation.
6. **Try offline KD.** Regenerate with `--topk 20` and train
   `--method logit --offline`: no teacher in memory, a fraction of the cost.

## Deliverable

A comparison table: GSM8K accuracy, GPU-hours, peak memory, and teacher
forwards for base, SFT-KD, logit-KD, GKD, and the teacher.

## Reference output (`python losses.py`, computed)

```
teacher: 55% mass near token 8, 45% near token 22
forward KL    mu= 13.8 sigma= 8.6  mass@mode1=0.35 mass@mode2=0.28 between=0.24
reverse KL    mu=  8.0 sigma= 2.0  mass@mode1=0.98 mass@mode2=0.00 between=0.01
GKD beta=0.9  mu=  8.0 sigma= 2.0  mass@mode1=0.98 mass@mode2=0.00 between=0.01
```
