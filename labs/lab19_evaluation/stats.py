"""
Lab 19: the statistics that make an eval result a decision.

  bootstrap_ci      percentile bootstrap CI for a mean score
  paired_bootstrap  P(model A > model B) on the SAME items (paired resampling)
  cohen_kappa       agreement between an LLM judge and human labels, corrected for chance
  min_items         items needed to detect a difference `delta` at ~95% confidence

The demo answers the question every eval owner gets: "with 300 items, how
small a difference can we trust?"

Usage:
    python stats.py
"""
from __future__ import annotations

import math
import random


def bootstrap_ci(scores, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(sum(rng.choice(scores) for _ in range(n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def paired_bootstrap(a, b, n_boot: int = 5000, seed: int = 0) -> float:
    """Fraction of resamples where A's mean beats B's; items resampled jointly."""
    rng = random.Random(seed)
    n, wins = len(a), 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        wins += sum(a[i] - b[i] for i in idx) > 0
    return wins / n_boot


def cohen_kappa(x, y) -> float:
    labels = sorted(set(x) | set(y))
    n = len(x)
    po = sum(i == j for i, j in zip(x, y)) / n
    pe = sum((x.count(l) / n) * (y.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe)


def min_items(p: float, delta: float, z: float = 1.96) -> int:
    """Unpaired normal approximation: n so that the 95% CI half-width of a
    difference of two accuracies near p is below delta."""
    return math.ceil(2 * (z ** 2) * p * (1 - p) / delta ** 2)


if __name__ == "__main__":
    rng = random.Random(1)
    n = 300
    a = [1 if rng.random() < 0.72 else 0 for _ in range(n)]
    # B shares most item difficulty with A (paired), slightly worse
    b = [x if rng.random() < 0.85 else (1 if rng.random() < 0.66 else 0) for x in a]
    ma, mb = sum(a) / n, sum(b) / n
    lo_a, hi_a = bootstrap_ci(a)
    lo_b, hi_b = bootstrap_ci(b)
    print(f"model A: {ma:.3f}  95% CI [{lo_a:.3f}, {hi_a:.3f}]")
    print(f"model B: {mb:.3f}  95% CI [{lo_b:.3f}, {hi_b:.3f}]")
    print(f"paired bootstrap P(A > B) = {paired_bootstrap(a, b):.3f}")
    human = [rng.choice([1, 2, 3, 4, 5]) for _ in range(100)]
    judge = [h if rng.random() < 0.7 else max(1, min(5, h + rng.choice([-1, 1]))) for h in human]
    print(f"judge-vs-human Cohen's kappa (1-5 scale, 100 items) = {cohen_kappa(human, judge):.3f}")
    for d in (0.10, 0.05, 0.03, 0.02):
        print(f"items needed to resolve a {d:.0%} difference at accuracy ~0.7: {min_items(0.7, d):,}")
