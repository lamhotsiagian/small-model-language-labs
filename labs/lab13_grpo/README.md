# Lab 13: GRPO on a 1.5B Model

**Chapter:** 13, Reasoning in Small Models
**Goal:** Train a 1.5B model with GRPO on GSM8K using verifiable rewards.
Track accuracy, response length, and reward over time. Deliver training
curves plus an analysis of emergent reasoning behaviour.

## Files

| File | Purpose |
|---|---|
| `rewards.py` | Correctness (answer checker), format (`<think>` + `####`), soft overlong penalty, GRPO and Dr. GRPO group advantages; `python rewards.py` demo |
| `train_grpo.py` | TRL `GRPOTrainer`: G=8, LR 1e-6, beta 0, optional vLLM generation, `--loss-type grpo | dr_grpo | dapo` |
| `analyze.py` | Reward / accuracy / length curves from `trainer_state.json`; reflection-marker frequency in sampled completions |
| `test_time.py` | Greedy vs self-consistency vs pass@N as N grows |

## Flow

1. **Understand the signal.** `python rewards.py` shows rewards and advantages for a
   group of four answers, and why an all-correct group gives zero gradient.
2. **Baseline.** `python ../lab08_distillation/evaluate.py --models Qwen/Qwen2.5-1.5B-Instruct`
   and `python test_time.py --model Qwen/Qwen2.5-1.5B-Instruct`.
3. **Train.** `python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 500 --use-vllm`
   (one 80 GB GPU; about 2-6 hours depending on generation length).
4. **Analyse.** `python analyze.py --run checkpoints/grpo`. Look for: reward rising,
   length first dropping (format learning) then rising (longer reasoning), and
   reflection markers appearing more often.
5. **Re-evaluate** on the GSM8K test set with greedy and self-consistency.
6. **Ablate.** Repeat with `--loss-type dr_grpo` and with the format reward removed;
   compare length growth and accuracy.

## Deliverable

Curves (reward, train accuracy, completion length), GSM8K test accuracy before
and after (greedy, SC@8, pass@8), and a one-page analysis of behaviour changes
with three annotated sample completions.

## Reference output (`python rewards.py`, computed)

```
GRPO advantages (group of 4):    [1.019, 0.679, -0.679, -1.019]
Dr. GRPO advantages (unscaled):  [0.6, 0.4, -0.4, -0.6]
all-correct group -> advantages: [0.0, 0.0, 0.0, 0.0] (no learning signal)
```
