# Lab 12: DPO vs ORPO vs SimPO

**Chapter:** 12, Preference Optimization and Alignment
**Goal:** Generate ~10K preference pairs with a judge model, then align the
Lab 11 SFT model with three methods. Evaluate win rate, verbosity, and IFEval.
Deliver a method comparison with qualitative failure analysis.

## Files

| File | Purpose |
|---|---|
| `pref_losses.py` | DPO, IPO, SimPO, ORPO, KTO written out; `python pref_losses.py` prints each loss on a pure length-difference pair |
| `make_pairs.py` | On-policy sampling (K=4) from the SFT model, rubric judging 1-10, best-vs-worst with margin filter, length-bias report |
| `train_pref.py` | TRL `DPOTrainer` (dpo/ipo), `CPOTrainer` (simpo), `ORPOTrainer` (orpo) with small-model defaults |
| `../lab11_sft/judge_pairwise.py` | Swap-consistent win rate vs the SFT model |

## Flow

1. **Read the objectives.** `python pref_losses.py`.
2. **Make pairs.** `python make_pairs.py --policy ../lab11_sft/checkpoints/sft --n 12000`.
   Check the printed length report: if chosen is longer in > 65% of pairs, rebalance
   (length-matched sampling or drop the longest decile) before training.
3. **Train three models** from the same SFT checkpoint on the same pairs:
   ```bash
   python train_pref.py --method dpo   --policy ../lab11_sft/checkpoints/sft
   python train_pref.py --method orpo  --policy ../lab11_sft/checkpoints/sft
   python train_pref.py --method simpo --policy ../lab11_sft/checkpoints/sft
   ```
4. **Evaluate.** For each: IFEval (`lm_eval ... --tasks ifeval --apply_chat_template`),
   swap-consistent win rate vs SFT (`judge_pairwise.py`), and mean response words.
   Also report a *length-controlled* win rate: win rate on the subset of prompts where
   the two responses differ in length by less than 20%.
5. **Failure analysis.** Read 30 losses per method. Tag each: verbosity, over-refusal,
   formatting drift, factual error, sycophancy. Count tags per method.

## Deliverable

Comparison table (IFEval, win rate, length-controlled win rate, mean words,
GPU-hours) and a tagged failure analysis per method.
