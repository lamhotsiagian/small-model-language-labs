"""
Lab 17, step 1: serving capacity and speculative-decoding math (computed).

Part A: how many concurrent sequences fit? After the weights, the rest of GPU
memory (times a utilisation factor) holds the KV cache. Concurrency at a given
context = KV budget / (KV bytes per token x context). This is the number that
PagedAttention lets you actually reach (no fragmentation).

Part B: speculative decoding expected speedup (Leviathan et al., 2023). With
a draft that proposes gamma tokens, per-token acceptance rate alpha, and draft
cost c (draft forward time / target forward time):

    E[tokens per target forward] = (1 - alpha^(gamma+1)) / (1 - alpha)
    speedup ~= E[tokens] / (1 + gamma * c)

Usage:
    python capacity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.budget import ZOO, kv_bytes_per_token, param_breakdown  # noqa: E402


def concurrency(spec, gpu_gb: float, context: int, util: float = 0.9, kv_dtype: str = "fp16",
                weight_bytes: float = 2.0) -> int:
    weights = param_breakdown(spec)["total"] * weight_bytes
    kv_budget = gpu_gb * 2 ** 30 * util - weights - 1.5 * 2 ** 30       # 1.5 GB activations/runtime
    return max(0, int(kv_budget // (kv_bytes_per_token(spec, kv_dtype) * context)))


def spec_speedup(alpha: float, gamma: int, c: float) -> tuple[float, float]:
    e_tokens = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    return e_tokens, e_tokens / (1 + gamma * c)


def main() -> None:
    print("A. Max concurrent sequences on one 24 GB GPU (bf16 weights, 90% utilisation)")
    print(f"{'model':<14}{'KV/tok':>8}{'@2K fp16':>10}{'@8K fp16':>10}{'@8K fp8':>10}")
    for name in ("Qwen2.5-0.5B", "Llama-3.2-1B", "Qwen2.5-1.5B", "SmolLM2-1.7B", "Llama-3.2-3B"):
        s = ZOO[name]
        print(f"{name:<14}{kv_bytes_per_token(s) / 1024:>6.0f}KB{concurrency(s, 24, 2048):>10}"
              f"{concurrency(s, 24, 8192):>10}{concurrency(s, 24, 8192, kv_dtype='fp8'):>10}")

    print("\nB. Speculative decoding: expected tokens per target step and speedup")
    print(f"{'alpha':>6}{'gamma':>6}{'c':>6}{'E[tokens]':>11}{'speedup':>9}")
    for alpha in (0.6, 0.7, 0.8, 0.9):
        for gamma, c in ((4, 0.05), (4, 0.15), (8, 0.05)):
            e, sp = spec_speedup(alpha, gamma, c)
            print(f"{alpha:>6.1f}{gamma:>6}{c:>6.2f}{e:>11.2f}{sp:>8.2f}x")


if __name__ == "__main__":
    main()
