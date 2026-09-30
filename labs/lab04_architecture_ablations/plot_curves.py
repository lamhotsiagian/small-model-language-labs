"""
Lab 4, step 4: plot validation-loss curves for every variant on one chart.

Usage:
    python plot_curves.py --results results/ablation.json --out results/loss_curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/ablation.json")
    ap.add_argument("--out", default="results/loss_curves.png")
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text())
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in res.items():
        toks = [h["tokens"] / 1e6 for h in r["history"]]
        ax.plot(toks, [h["val_loss"] for h in r["history"]], label=f"{name} (L={r['layers']})")
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("validation loss (nats/token)")
    ax.set_title("Architecture ablations at matched parameters and tokens")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print(f"[lab04] wrote {a.out}")


if __name__ == "__main__":
    main()
