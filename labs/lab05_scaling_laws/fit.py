"""
Lab 5, step 3: fit a Chinchilla-style scaling law and predict a larger model.

    L(N, D) = E + A / N^alpha + B / D^beta

  E        irreducible loss (entropy of the data under the tokenizer)
  A/N^a    error from finite model capacity
  B/D^b    error from finite data

Fitting follows Hoffmann et al. (2022): minimise a Huber loss between
log-predicted and log-observed loss, optimising in log-parameter space, from
a grid of initialisations (the objective is non-convex).

Usage:
    python fit.py --runs results/sweep.json --holdout-n 250e6
    python fit.py --selftest       # recover known parameters from synthetic runs
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import huber

CHINCHILLA = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)   # Hoffmann et al. (2022)


def predict(p: dict, N, D):
    return p["E"] + p["A"] / np.power(N, p["alpha"]) + p["B"] / np.power(D, p["beta"])


def fit(N: np.ndarray, D: np.ndarray, L: np.ndarray, delta: float = 1e-3) -> dict:
    """Huber regression in log space with log-parameterised A, B, E."""
    logN, logD, logL = np.log(N), np.log(D), np.log(L)

    def objective(theta):
        a, b, e, alpha, beta = theta
        # log-sum-exp form of log(A/N^alpha + B/D^beta + E) is numerically stable
        terms = np.stack([a - alpha * logN, b - beta * logD, np.full_like(logN, e)])
        pred = np.logaddexp.reduce(terms, axis=0)
        return huber(delta, pred - logL).sum()

    best = None
    for a0, b0, e0, al0, be0 in itertools.product([0, 5, 10], [0, 5, 10], [-1, 0.5],
                                                  [0.2, 0.5], [0.2, 0.5]):
        r = minimize(objective, x0=[a0, b0, e0, al0, be0], method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    a, b, e, alpha, beta = best.x
    return {"E": float(np.exp(e)), "A": float(np.exp(a)), "B": float(np.exp(b)),
            "alpha": float(alpha), "beta": float(beta), "objective": float(best.fun)}


def compute_optimal(p: dict, C: float) -> tuple[float, float]:
    """Minimise L(N, D) subject to 6ND = C (closed form for this law)."""
    G = (p["alpha"] * p["A"] / (p["beta"] * p["B"])) ** (1 / (p["alpha"] + p["beta"]))
    a = p["beta"] / (p["alpha"] + p["beta"])
    N = G * (C / 6) ** a
    return N, C / (6 * N)


def selftest(noise: float = 0.01, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    Ns = np.array([5e6, 15e6, 40e6, 100e6])
    Ds = np.array([0.2e9, 0.6e9, 2e9])
    N, D = (a.ravel() for a in np.meshgrid(Ns, Ds))
    L = predict(CHINCHILLA, N, D) * np.exp(rng.normal(0, noise, N.shape))
    p = fit(N, D, L)
    print("parameter   true      fitted")
    for k in ("E", "A", "B", "alpha", "beta"):
        print(f"{k:<9}{CHINCHILLA[k]:>8.3f}{p[k]:>12.3f}")
    for n_new, d_new in [(250e6, 2e9), (250e6, 5e9)]:
        true, pred = predict(CHINCHILLA, n_new, d_new), predict(p, n_new, d_new)
        print(f"predict N={n_new/1e6:.0f}M D={d_new/1e9:.0f}B: true {true:.4f}  "
              f"fitted {pred:.4f}  error {100 * (pred - true) / true:+.2f}%")
    N_opt, D_opt = compute_optimal(CHINCHILLA, 6 * 1e9 * 20e9)
    print(f"compute-optimal at C=6*1B*20B: N={N_opt/1e9:.2f}B D={D_opt/1e9:.1f}B "
          f"(D/N={D_opt/N_opt:.1f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/sweep.json")
    ap.add_argument("--holdout-n", type=float, default=250e6)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    runs = json.loads(Path(a.runs).read_text())
    train = [r for r in runs if r["params"] < a.holdout_n * 0.9]
    test = [r for r in runs if r["params"] >= a.holdout_n * 0.9]
    N = np.array([r["params"] for r in train], float)
    D = np.array([r["tokens"] for r in train], float)
    L = np.array([r["val_loss"] for r in train], float)
    p = fit(N, D, L)
    print(json.dumps(p, indent=2))
    for r in test:
        pred = predict(p, r["params"], r["tokens"])
        print(f"held-out N={r['params']/1e6:.0f}M D={r['tokens']/1e9:.2f}B: "
              f"observed {r['val_loss']:.4f} predicted {pred:.4f} "
              f"error {100 * (pred - r['val_loss']) / r['val_loss']:+.2f}%")
    Path("results/fit.json").write_text(json.dumps(p, indent=2))


if __name__ == "__main__":
    main()
