"""
Lab 20, step 5: input drift monitoring with the Population Stability Index.

Compares a reference window (the traffic the model was evaluated on) with a
live window on cheap, content-free features:
  prompt length (tokens), fraction non-ASCII (language shift), digits ratio,
  and the router's predicted category mix.

PSI = sum_bins (live - ref) * ln(live / ref). Rules of thumb:
  < 0.1 stable, 0.1-0.25 watch, > 0.25 investigate (and re-run the domain eval
  on a sample of live traffic).

Usage:
    python drift_monitor.py --demo
    python drift_monitor.py --ref logs/ref.jsonl --live logs/live.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random


def psi(ref, live, bins: int = 10) -> float:
    lo, hi = min(ref + live), max(ref + live)
    width = (hi - lo) / bins or 1.0

    def hist(xs):
        h = [0] * bins
        for x in xs:
            h[min(bins - 1, int((x - lo) / width))] += 1
        return [(c + 0.5) / (len(xs) + 0.5 * bins) for c in h]      # smoothed
    r, l = hist(ref), hist(live)
    return sum((b - a) * math.log(b / a) for a, b in zip(r, l))


def features(text: str) -> dict:
    n = max(1, len(text))
    return {"length": len(text.split()), "non_ascii": sum(ord(c) > 127 for c in text) / n,
            "digits": sum(c.isdigit() for c in text) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref"); ap.add_argument("--live")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        rng = random.Random(0)
        ref = [" ".join("w" for _ in range(int(rng.gauss(40, 10)) or 1)) for _ in range(2000)]
        same = [" ".join("w" for _ in range(int(rng.gauss(40, 10)) or 1)) for _ in range(2000)]
        longer = [" ".join("w" for _ in range(int(rng.gauss(70, 20)) or 1)) for _ in range(2000)]
        fr = [features(t)["length"] for t in ref]
        print(f"PSI length, same distribution:   {psi(fr, [features(t)['length'] for t in same]):.3f}")
        print(f"PSI length, prompts got longer:  {psi(fr, [features(t)['length'] for t in longer]):.3f}")
        return
    ref = [features(json.loads(l)["user"]) for l in open(a.ref)]
    live = [features(json.loads(l)["user"]) for l in open(a.live)]
    for k in ref[0]:
        v = psi([r[k] for r in ref], [x[k] for x in live])
        flag = "stable" if v < 0.1 else "watch" if v < 0.25 else "INVESTIGATE"
        print(f"{k:<10} PSI {v:.3f}  {flag}")


if __name__ == "__main__":
    main()
