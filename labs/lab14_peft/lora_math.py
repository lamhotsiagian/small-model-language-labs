"""
Lab 14, step 1: LoRA from first principles, plus a memory calculator.

  * LoRALinear: y = W x + (alpha / r) * B A x, with A ~ small random, B = 0,
    so training starts EXACTLY at the base model (Hu et al., 2022)
  * rsLoRA scaling alpha / sqrt(r) keeps update magnitude stable as r grows
    (Kalajdzievski, 2023)
  * merge(): W' = W + (alpha / r) B A, after which inference cost is zero
  * budget(): trainable params and training memory for full FT, LoRA, QLoRA
    on a Llama-family config

Usage:
    python lora_math.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.budget import ZOO, param_breakdown  # noqa: E402


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16, rslora: bool = False) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                              # frozen base weight
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zero -> identical at step 0
        self.scale = alpha / math.sqrt(r) if rslora else alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.t() @ self.B.t())

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=False)
        merged.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
        return merged


def budget(spec, r: int, targets=("q", "k", "v", "o", "gate", "up", "down")) -> dict:
    d, hd, L = spec.d_model, spec.hd, spec.n_layers
    shapes = {"q": (d, spec.n_heads * hd), "k": (d, spec.n_kv_heads * hd), "v": (d, spec.n_kv_heads * hd),
              "o": (spec.n_heads * hd, d), "gate": (d, spec.d_ff), "up": (d, spec.d_ff), "down": (spec.d_ff, d)}
    lora = L * sum(r * (i + o) for t, (i, o) in shapes.items() if t in targets)
    n = param_breakdown(spec)["total"]
    gb = 2 ** 30
    return {"total": n, "lora_params": lora, "share": lora / n,
            # full FT: bf16 weights + bf16 grads + fp32 master + Adam m, v
            "full_ft_gb": n * 16 / gb,
            # LoRA: bf16 frozen base + (bf16 adapter + fp32 master/m/v + grad) for adapter
            "lora_gb": (n * 2 + lora * 16) / gb,
            # QLoRA: NF4 linear layers (~4.127 bits/param with double-quantised constants);
            # bitsandbytes keeps the (tied) embedding in bf16
            "qlora_gb": ((n - spec.vocab * d) * (4.127 / 8) + spec.vocab * d * 2 + lora * 16) / gb}


def main() -> None:
    torch.manual_seed(0)
    base = nn.Linear(512, 512, bias=False)
    lora = LoRALinear(base, r=8, alpha=16)
    x = torch.randn(4, 512)
    print(f"step-0 output difference vs base: {(lora(x) - base(x)).abs().max():.1e}  (B = 0)")
    with torch.no_grad():
        lora.B.normal_(0, 0.02)                                   # pretend we trained
    print(f"merged vs unmerged max diff:      {(lora.merge()(x) - lora(x)).abs().max():.1e}")

    spec = ZOO["Llama-3.2-3B"]
    print(f"\n{spec.name}: training memory before activations (weights + grads + optimizer)")
    print(f"{'method':<22}{'trainable':>12}{'share':>8}{'memory':>10}")
    full = budget(spec, 8)
    print(f"{'full fine-tune':<22}{full['total'] / 1e6:>10.0f}M{1:>8.0%}{full['full_ft_gb']:>8.1f}GB")
    for r in (8, 64):
        b = budget(spec, r)
        print(f"{f'LoRA r={r} (all linear)':<22}{b['lora_params'] / 1e6:>10.1f}M{b['share']:>8.2%}{b['lora_gb']:>8.1f}GB")
    for r in (8, 64):
        b = budget(spec, r)
        print(f"{f'QLoRA r={r}':<22}{b['lora_params'] / 1e6:>10.1f}M{b['share']:>8.2%}{b['qlora_gb']:>8.1f}GB")
    b = budget(spec, 8, targets=("q", "v"))
    print(f"{'LoRA r=8 (q,v only)':<22}{b['lora_params'] / 1e6:>10.1f}M{b['share']:>8.2%}{b['lora_gb']:>8.1f}GB")


if __name__ == "__main__":
    main()
