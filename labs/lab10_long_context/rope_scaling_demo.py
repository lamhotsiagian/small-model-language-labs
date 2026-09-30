"""
Lab 10, step 1: what RoPE scaling methods actually do to the frequencies.

For a head_dim of 64 and base theta = 10,000 (trained at 8K), print the
wavelength (tokens per full rotation) of selected dimension pairs under:

  original   theta_i = base^(-2i/d)
  PI         positions divided by s   -> every wavelength stretched by s
  NTK-aware  base raised to base * s^(d/(d-2)) -> low frequencies stretched ~s,
             high frequencies almost untouched
  YaRN       NTK-by-parts: pairs that rotate many times per original context
             are left alone, pairs that rotate < 1 time are interpolated by s,
             a linear ramp in between; plus attention temperature 0.1 ln s + 1

The key column is "rotations in 8K": pairs with fewer than ~1 rotation in
the original context have never seen positions past 8K and MUST be
interpolated; pairs with many rotations encode local order and should not be.

Usage:
    python rope_scaling_demo.py --factor 4
"""
from __future__ import annotations

import argparse
import math

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.model import SLMConfig, rope_frequencies, yarn_mscale


def wavelengths(cfg: SLMConfig):
    """Tokens per full rotation for each dimension pair, read off the RoPE table."""
    import torch
    cos, sin = rope_frequencies(cfg, 2)                 # row 1 = rotation angle per token
    ang = torch.atan2(sin[1], cos[1])
    return (2 * math.pi / ang).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("--orig", type=int, default=8192)
    ap.add_argument("--head-dim", type=int, default=64)
    a = ap.parse_args()
    base = dict(vocab_size=10, d_model=a.head_dim, n_heads=1, n_kv_heads=1, max_seq_len=a.orig)
    variants = {
        "original": SLMConfig(**base),
        "PI": SLMConfig(**base, rope_scaling={"type": "linear", "factor": a.factor}),
        "NTK": SLMConfig(**base, rope_scaling={"type": "ntk", "factor": a.factor}),
        "YaRN": SLMConfig(**base, rope_scaling={"type": "yarn", "factor": a.factor, "orig_max": a.orig}),
    }
    wl = {k: wavelengths(v) for k, v in variants.items()}
    print(f"head_dim={a.head_dim} base=10000 trained at {a.orig} tokens, scale factor s={a.factor:g}\n")
    print(f"{'pair i':>6}{'rotations in 8K':>17}" + "".join(f"{k:>11}" for k in wl))
    for i in (0, 4, 8, 12, 16, 20, 24, 28, 31):
        rot = a.orig / wl["original"][i]
        print(f"{i:>6}{rot:>17.2f}" + "".join(f"{wl[k][i]:>11.0f}" for k in wl))
    print(f"\nYaRN attention temperature (mscale) = {yarn_mscale(variants['YaRN']):.4f}")


if __name__ == "__main__":
    main()
