"""
Lab 2, step 1: scaled dot-product attention from first principles.

Implements causal multi-head attention with explicit tensor algebra, checks it
against PyTorch's fused kernel, and demonstrates two facts from Chapter 2:

  1. Why we divide by sqrt(d_k): without it the logit variance grows with d_k
     and softmax saturates (near one-hot attention, vanishing gradients).
  2. Causal masking = adding -inf above the diagonal before the softmax, so
     position t can only attend to positions <= t.

Usage:
    python attention_math.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def naive_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q, k, v: [batch, heads, seq, d_k]. Returns [batch, heads, seq, d_k]."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)          # [B, H, T, T]
    T = q.shape[-2]
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))           # hide the future
    weights = scores.softmax(dim=-1)                             # rows sum to 1
    return weights @ v


def softmax_entropy(scores: torch.Tensor) -> float:
    p = scores.softmax(-1)
    return float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean())


def main() -> None:
    torch.manual_seed(0)
    B, H, T, d_k = 2, 4, 16, 64
    q, k, v = (torch.randn(B, H, T, d_k) for _ in range(3))

    ours = naive_causal_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"max |naive - fused| = {(ours - ref).abs().max():.2e}")

    # Variance of q.k grows linearly with d_k; the 1/sqrt(d_k) factor undoes it.
    print("\n d_k   var(q.k)  var(q.k/sqrt(d_k))  entropy(unscaled)  entropy(scaled)")
    for d in (16, 64, 256, 1024):
        qq, kk = torch.randn(4096, d), torch.randn(4096, d)
        dots = (qq * kk).sum(-1)
        s_raw = torch.randn(256, 128, d) @ torch.randn(256, d, 128)
        s_scaled = s_raw / math.sqrt(d)
        print(f"{d:5d} {dots.var():9.1f} {(dots / math.sqrt(d)).var():14.2f}"
              f" {softmax_entropy(s_raw):17.3f} {softmax_entropy(s_scaled):15.3f}")
    print(f"\n(max possible entropy over 128 keys = ln 128 = {math.log(128):.3f})")


if __name__ == "__main__":
    main()
