# Lab 5: Mini Scaling Law Fit

**Chapter:** 5, Scaling Laws and Compute Budgeting
**Goal:** Train models at ~5M, 15M, 40M, and 100M parameters over three token
budgets, fit L(N, D) = E + A/N^alpha + B/D^beta, predict the loss of a 250M
model, then train that model and report the prediction error.

## Files

| File | Purpose |
|---|---|
| `sweep.py` | Size solver (d/L aspect ~48, 64-dim heads), WSD branching (one stable run per size, one decay branch per budget), muP-lite LR scaling |
| `fit.py` | Huber-in-log-space fit with multi-start L-BFGS (Hoffmann et al., 2022), closed-form compute-optimal N/D, `--selftest` on synthetic ground truth |
| `inference_aware.py` | Minimises training + lifetime inference FLOPs at a target loss |

## Flow

1. **Self-test the fitter.** `python fit.py --selftest` generates runs from the
   published Chinchilla law with 1% noise and fits them. Note that individual
   parameters are poorly identified from a small grid while interpolated
   predictions stay accurate: trust predictions, not coefficients.
2. **Run the sweep.** `python sweep.py --preset full --device cuda` trains
   4 sizes x 3 budgets (0.2B, 0.6B, 2B tokens) plus the 250M hold-out. WSD
   branching makes the 3 budgets cost ~1.2x a single run per size.
3. **Fit.** `python fit.py --runs results/sweep.json --holdout-n 250e6` fits on
   the small runs only and predicts the 250M runs.
4. **Verify.** Compare predicted and observed held-out loss. A good fit is within
   1-2% in loss; report the error and plot residuals against N and D.
5. **Size a product model.** `python inference_aware.py --fit results/fit.json`
   shows how the optimal (N, D) moves as lifetime inference volume grows.

## Deliverable

Fitted curve, prediction error report, and the inference-aware sizing table.

## Reference output (self-test, computed)

```
predict N=250M D=2B: true 3.2786  fitted 3.2876  error +0.27%
predict N=250M D=5B: true 3.0474  fitted 2.9944  error -1.74%
```
