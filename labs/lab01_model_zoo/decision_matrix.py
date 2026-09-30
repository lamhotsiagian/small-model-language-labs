"""
Lab 1, step 4: turn zoo_results.json into a decision matrix.

For each task type the script scores every model with a weighted utility

    utility = w_q * quality  -  w_l * norm(latency)  -  w_m * norm(memory)

subject to hard constraints (a quality floor and a memory ceiling). The
smallest model that clears the floor wins ties, because in production the
cheaper model that is good enough beats the better model you cannot afford.

Also prints a self-hosting vs API cost comparison (Chapter 1, section 1.5).

Usage:
    python decision_matrix.py --results results/zoo_results.json \
        --quality-floor 0.8 --memory-ceiling-mb 2500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalise(values):
    lo, hi = min(values), max(values)
    return [0.0 if hi == lo else (v - lo) / (hi - lo) for v in values]


def decide(results, task_type, floor, ceiling_mb, w_q=1.0, w_l=0.3, w_m=0.2):
    rows = [(r["model"], r["per_task"][task_type], r["weights_mb"]) for r in results
            if task_type in r["per_task"]]
    lat = normalise([p["ttft_ms_p50"] + 64 * p["tpot_ms_p50"] for _, p, _ in rows])
    mem = normalise([m for _, _, m in rows])
    scored = []
    for (name, p, m), l, mm in zip(rows, lat, mem):
        feasible = p["quality"] >= floor and m <= ceiling_mb
        u = w_q * p["quality"] - w_l * l - w_m * mm
        scored.append((feasible, round(u, 3), -m, name, p["quality"]))
    scored.sort(reverse=True)
    return scored


def tco_per_million_tokens(gpu_hourly_usd: float, tokens_per_s: float,
                           utilisation: float = 0.5) -> float:
    """Self-hosted cost per 1M output tokens at a given average utilisation."""
    tokens_per_hour = tokens_per_s * 3600 * utilisation
    return gpu_hourly_usd / tokens_per_hour * 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/zoo_results.json")
    ap.add_argument("--quality-floor", type=float, default=0.8)
    ap.add_argument("--memory-ceiling-mb", type=float, default=2500)
    ap.add_argument("--out", default="results/decision_matrix.md")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    tasks = sorted({t for r in results for t in r["per_task"]})
    lines = ["| Task | Recommended | Quality | Runner-up | Notes |", "|---|---|---|---|---|"]
    for t in tasks:
        ranked = decide(results, t, args.quality_floor, args.memory_ceiling_mb)
        best = next((r for r in ranked if r[0]), None)
        if best is None:
            lines.append(f"| {t} | none meets floor | - | {ranked[0][3]} | escalate to larger model or fine-tune |")
            continue
        runner = next((r for r in ranked if r[0] and r is not best), None)
        lines.append(f"| {t} | {best[3]} | {best[4]:.2f} | {runner[3] if runner else '-'} | utility {best[1]} |")
    md = "\n".join(lines)
    Path(args.out).write_text(md + "\n")
    print(md)

    print("\nSelf-hosted cost per 1M output tokens (50% utilisation):")
    for gpu, price, tps in [("L4 24GB, 1B model, batched", 0.80, 2500),
                            ("A10G 24GB, 3B model, batched", 1.20, 1800),
                            ("CPU 16 vCPU, 0.5B Q4, batch 1", 0.60, 40)]:
        print(f"  {gpu:<32} ${tco_per_million_tokens(price, tps):6.2f}")


if __name__ == "__main__":
    main()
