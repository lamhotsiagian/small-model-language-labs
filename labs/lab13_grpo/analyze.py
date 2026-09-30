"""
Lab 13, step 4: training curves and emergent-behaviour analysis.

Reads TRL's trainer_state.json and plots reward, correctness, completion
length, and (when logged) KL over steps. Then scans sampled completions for
reflection markers ("wait", "let me check", "verify") whose frequency often
rises during RL on math, and reports it per checkpoint.

Usage:
    python analyze.py --run checkpoints/grpo --samples results/samples_step*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MARKERS = re.compile(r"\b(wait|let me (?:check|verify|re-?check)|double[- ]check|hmm)\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="checkpoints/grpo")
    ap.add_argument("--samples", default="results/samples_step*.jsonl")
    a = ap.parse_args()
    logs = json.loads(Path(a.run, "trainer_state.json").read_text())["log_history"]
    keys = {"reward": "reward", "correct": "rewards/correctness_reward/mean",
            "length": "completions/mean_length", "kl": "kl"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (title, k) in zip(axes, [("mean reward", keys["reward"]), ("accuracy (train)", keys["correct"]),
                                     ("completion length (tokens)", keys["length"])]):
        pts = [(r["step"], r[k]) for r in logs if k in r]
        if pts:
            ax.plot(*zip(*pts))
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    fig.tight_layout()
    Path("results").mkdir(exist_ok=True)
    fig.savefig("results/grpo_curves.png", dpi=160)
    for f in sorted(glob.glob(a.samples)):
        rows = [json.loads(l) for l in open(f)]
        rate = sum(bool(MARKERS.search(r["completion"])) for r in rows) / max(1, len(rows))
        print(f"{Path(f).name}: reflection markers in {rate:.1%} of {len(rows)} samples")
    print("[lab13] wrote results/grpo_curves.png")


if __name__ == "__main__":
    main()
