"""
Lab 10, step 5: memory profile vs context length, and what KV compression buys.

Prints, for a model config, the weights + KV cache at 4K-32K for:
  fp16 KV, int8 KV, int4 KV, a StreamingLLM window (4 sinks + 2,044 recent),
  and a GQA counterfactual (same model with 4x fewer KV heads).

Usage:
    python kv_profile.py --arch SmolLM2-1.7B
    python kv_profile.py --arch Qwen2.5-1.5B
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.budget import BYTES, ZOO, kv_bytes_per_token, param_breakdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="SmolLM2-1.7B", choices=list(ZOO))
    a = ap.parse_args()
    spec = ZOO[a.arch]
    w_gb = param_breakdown(spec)["total"] * BYTES["bf16"] / 2 ** 30
    gqa = replace(spec, n_kv_heads=max(1, spec.n_kv_heads // 4))
    print(f"{spec.name}: weights {w_gb:.2f} GB (bf16), KV heads {spec.n_kv_heads}, "
          f"KV/token {kv_bytes_per_token(spec) / 1024:.0f} KB (fp16)\n")
    print(f"{'context':>8}{'fp16 KV':>10}{'int8 KV':>10}{'int4 KV':>10}{'stream 2K':>11}{'GQA/4 fp16':>12}")
    for T in (4096, 8192, 16384, 32768):
        row = [kv_bytes_per_token(spec, d) * T / 2 ** 30 for d in ("fp16", "int8", "int4")]
        stream = kv_bytes_per_token(spec) * min(T, 2048) / 2 ** 30
        g = kv_bytes_per_token(gqa) * T / 2 ** 30
        print(f"{T:>8}" + "".join(f"{v:>9.2f}G" for v in row) + f"{stream:>10.2f}G{g:>11.2f}G")


if __name__ == "__main__":
    main()
