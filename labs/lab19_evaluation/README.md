# Lab 19: Custom Evaluation Suite

**Chapter:** 19, Evaluation and Benchmarking
**Goal:** Build a 300-item domain evaluation (with rubric and judge prompt),
run standard harness benchmarks, and evaluate all models from previous labs.
Deliver a reproducible eval repo and a leaderboard.

## Files

| File | Purpose |
|---|---|
| `eval_spec.yaml` | Versioned suite definition: categories and scorers, quotas, frozen decoding, judge and calibration requirements, canary |
| `build_items.py` | Schema / quota / duplicate validation and a stratified dev/test split; `--demo` shows the item schema |
| `stats.py` | Bootstrap CIs, paired bootstrap, Cohen's kappa, items needed for a given resolution |
| `run_suite.py` | Runs every model with its own template, scores per category (exact, F1, AST, rubric judge, refusal checks), writes the leaderboard |
| `harness.sh` | lm-evaluation-harness (vLLM backend) with identical settings for every model |

## Flow

1. **Write the spec** (`eval_spec.yaml`) before writing items: categories, scorers, quotas,
   decoding settings, and the judge acceptance bar.
2. **Author items.** 300 items across the six categories; two annotators per item;
   resolve disagreements; keep policy-sensitive items in an access-controlled file.
   `python build_items.py --items data/items.jsonl`.
3. **Calibrate the judge.** Label 60 rubric items by hand, score them with the judge,
   and compute Cohen's kappa with `stats.cohen_kappa`. Do not use the judge below 0.6.
4. **Standard benchmarks.** `bash harness.sh <model> ...` for every model from Labs 7-18.
5. **Domain suite.** `python run_suite.py --models <all models>` -> `results/leaderboard.md`.
6. **Decide with statistics.** `python stats.py` shows how large a difference 300 items can
   resolve; use `paired_bootstrap` for every "A beats B" claim.
7. **Freeze.** Tag the repo: suite version, item hashes, judge model revision, decoding settings.

## Deliverable

A tagged eval repository and a leaderboard with per-category scores, bootstrap
CIs, harness results, and quality-per-GB for every model.

## Reference output (`python stats.py`, computed)

```
model A: 0.717  95% CI [0.667, 0.767]
model B: 0.703  95% CI [0.650, 0.757]
paired bootstrap P(A > B) = 0.766
items needed to resolve a 5% difference at accuracy ~0.7: 646
```
