"""
Lab 14, step 4: cost-quality frontier (peak VRAM vs test accuracy, bubble = minutes).

Usage:
    python plot_frontier.py
"""
from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    rows = [json.load(open(f)) for f in sorted(glob.glob("results/*.json"))]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r in rows:
        ax.scatter(r["peak_gb"], r["test_acc"], s=30 + 6 * r["train_minutes"], alpha=0.7)
        ax.annotate(r["run"], (r["peak_gb"], r["test_acc"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("peak GPU memory (GB)"); ax.set_ylabel("test accuracy")
    ax.set_title("PEFT cost-quality frontier (bubble size = training minutes)")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("results/frontier.png", dpi=160)
    print("| run | trainable | peak GB | minutes | test acc |\n|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r["peak_gb"]):
        print(f"| {r['run']} | {r['trainable_params'] / 1e6:.1f}M | {r['peak_gb']} | {r['train_minutes']} | {r['test_acc']} |")


if __name__ == "__main__":
    main()
