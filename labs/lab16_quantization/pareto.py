"""
Lab 16, step 5: quality-vs-size Pareto chart and a recommended default.

Input: results/ladder.csv with columns
    name,size_mb,ppl,task_acc,tok_per_s
(fill from gguf_ladder.sh, lm-eval runs on the GPU formats, and vLLM/llama-bench).

Rule for the recommendation (Chapter 16, section 16.5): the SMALLEST artifact
whose task accuracy is within `--tolerance` points of the 16-bit baseline and
whose perplexity is within `--ppl-tolerance` (relative).

Usage:
    python pareto.py --csv results/ladder.csv --tolerance 1.0 --ppl-tolerance 0.05
"""
from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/ladder.csv")
    ap.add_argument("--baseline", default="fp16")
    ap.add_argument("--tolerance", type=float, default=1.0, help="task-accuracy points")
    ap.add_argument("--ppl-tolerance", type=float, default=0.05)
    a = ap.parse_args()
    rows = [{k: (v if k == "name" else float(v)) for k, v in r.items()} for r in csv.DictReader(open(a.csv))]
    base = next(r for r in rows if r["name"] == a.baseline)
    rows.sort(key=lambda r: r["size_mb"])
    front, best = [], -1
    for r in rows:                                      # Pareto: nothing smaller is at least as accurate
        if r["task_acc"] > best:
            front.append(r); best = r["task_acc"]
    ok = [r for r in rows if base["task_acc"] - r["task_acc"] <= a.tolerance
          and r["ppl"] <= base["ppl"] * (1 + a.ppl_tolerance)]
    rec = ok[0] if ok else base
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter([r["size_mb"] for r in rows], [r["task_acc"] for r in rows])
    ax.plot([r["size_mb"] for r in front], [r["task_acc"] for r in front], ls="--")
    for r in rows:
        ax.annotate(r["name"], (r["size_mb"], r["task_acc"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("artifact size (MB)"); ax.set_ylabel("task accuracy"); ax.grid(alpha=0.3)
    ax.set_title("Quantization ladder: quality vs size"); fig.tight_layout()
    fig.savefig("results/pareto.png", dpi=160)
    print(f"Pareto front: {[r['name'] for r in front]}")
    print(f"recommended default: {rec['name']} ({rec['size_mb']:.0f} MB, acc {rec['task_acc']}, ppl {rec['ppl']})")


if __name__ == "__main__":
    main()
