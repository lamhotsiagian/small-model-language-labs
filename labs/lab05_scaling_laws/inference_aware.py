"""
Lab 5, step 5: inference-aware sizing (why overtraining SLMs is rational).

Given a scaling law L(N, D) and an expected lifetime inference volume D_inf
(tokens served), find the model size N and training tokens D that reach a
target loss with minimum TOTAL compute:

    total FLOPs = 6 N D (training)  +  2 N D_inf (inference)

With D_inf = 0 this is the Chinchilla optimum. As D_inf grows, the optimum
shifts to smaller N trained on many more tokens (Sardana & Frankle, 2024).

Usage:
    python inference_aware.py                     # uses the Chinchilla fit
    python inference_aware.py --fit results/fit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fit import CHINCHILLA, predict


def tokens_for_loss(p: dict, N: float, target: float) -> float:
    """Solve L(N, D) = target for D (inf if N is too small to reach it)."""
    gap = target - p["E"] - p["A"] / N ** p["alpha"]
    return np.inf if gap <= 0 else (p["B"] / gap) ** (1 / p["beta"])


def optimise(p: dict, target: float, d_inf: float):
    best = None
    for N in np.logspace(8, 11, 3000):
        D = tokens_for_loss(p, N, target)
        if not np.isfinite(D):
            continue
        total = 6 * N * D + 2 * N * d_inf
        if best is None or total < best[0]:
            best = (total, N, D)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit")
    ap.add_argument("--ref-n", type=float, default=7e9, help="reference Chinchilla-optimal size")
    a = ap.parse_args()
    p = json.loads(Path(a.fit).read_text()) if a.fit else CHINCHILLA
    target = float(predict(p, a.ref_n, 20 * a.ref_n))
    print(f"target loss = Chinchilla-optimal {a.ref_n/1e9:.0f}B at 20 tok/param: {target:.4f}\n")
    print(f"{'lifetime inference':>20}{'N*':>9}{'D*':>10}{'D*/N*':>8}{'train share':>13}")
    for d_inf in [0, 1e11, 1e12, 1e13, 1e14]:
        total, N, D = optimise(p, target, d_inf)
        print(f"{d_inf:>20.0e}{N/1e9:>8.2f}B{D/1e12:>9.2f}T{D/N:>8.0f}{6*N*D/total:>12.0%}")


if __name__ == "__main__":
    main()
