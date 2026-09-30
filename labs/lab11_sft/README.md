# Lab 11: Build an Instruct SLM

**Chapter:** 11, Supervised Fine-Tuning (SFT)
**Goal:** Fine-tune a base 0.5-1.5B model on a curated 50K instruction set
using TRL; compare against the official instruct version on IFEval and
MT-Bench-style prompts; deliver a model card and an evaluation report.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Pool smoltalk / Tulu 3 / Magpie, clean, exact + MinHash dedup, decontaminate, cap categories, write prompt/completion format |
| `magpie_synth.py` | Optional self-synthesis of instructions from an aligned model's pre-query template |
| `train_sft.py` | TRL `SFTTrainer`: packing, completion-only loss, cosine LR; `--check-masking` shows trained vs masked tokens |
| `judge_pairwise.py` | Swap-consistent pairwise judging vs the official instruct model |
| `MODEL_CARD_TEMPLATE.md` | Fill in for the deliverable |

## Flow

1. **Build data.** `python build_dataset.py --n 50000` (prints drop counts per stage).
2. **Check the template and masking.** `python train_sft.py --check-masking --base Qwen/Qwen2.5-0.5B`.
   Only assistant tokens (including the end-of-turn token) may say TRAIN.
3. **LR sweep.** Train 1 epoch at `--lr 1e-5`, `2e-5`, `5e-5`; keep the best by held-out loss
   and a 50-prompt IFEval subset.
4. **Full run.** `python train_sft.py --lr <best> --epochs 2` (one 24 GB GPU for 0.5B; 80 GB for 1.5B).
5. **Evaluate.**
   ```bash
   lm_eval --model hf --model_args pretrained=checkpoints/sft --tasks ifeval --apply_chat_template
   lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct --tasks ifeval --apply_chat_template
   python judge_pairwise.py --a checkpoints/sft --b Qwen/Qwen2.5-0.5B-Instruct
   ```
6. **Forgetting check.** Run HellaSwag / ARC / GSM8K before and after SFT. If they
   drop, re-run with `--general-mix` (Chapter 11, section 11.5).
7. **Write the model card** from `MODEL_CARD_TEMPLATE.md`.

## Deliverable

Model card and evaluation report: IFEval strict/loose, swap-consistent win rate
vs the official instruct model, mean response length, and the forgetting check.
