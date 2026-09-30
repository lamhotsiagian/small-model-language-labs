"""
Lab 16, step 1: quantization fundamentals, measured on a realistic weight matrix.

We build a 2048 x 2048 "weight" with the two properties that make LLM
quantization hard: roughly normal values, plus a few OUTLIER input channels
with much larger magnitude (Dettmers et al., 2022). Then we quantize it with
the schemes of Chapter 16, section 16.1, and report reconstruction error
(relative Frobenius error) and effective bits per weight including scales.

  symmetric per-tensor     one scale for the whole matrix
  asymmetric per-tensor    scale + zero point
  symmetric per-channel    one scale per output row
  symmetric per-group g    one scale per g consecutive weights in a row (GPTQ/AWQ/GGUF style)
  NF4 per-group 64         normal-quantile levels (QLoRA)
  + an output-error view:  || XW^T - XW_q^T || with activations that also have outliers

Usage:
    python quant_math.py
"""
from __future__ import annotations

import torch

torch.manual_seed(0)


def make_weight(n: int = 2048, outlier_cols: int = 8, scale: float = 20.0) -> torch.Tensor:
    w = torch.randn(n, n) * 0.02
    cols = torch.randperm(n)[:outlier_cols]
    w[:, cols] *= scale                                   # a few input channels carry large weights
    return w


def q_sym(w, bits, dim=None, group=None):
    qmax = 2 ** (bits - 1) - 1
    if group:
        shp = w.shape
        wg = w.reshape(shp[0], -1, group)
        s = wg.abs().amax(-1, keepdim=True) / qmax
        return (torch.clamp(torch.round(wg / s), -qmax - 1, qmax) * s).reshape(shp)
    s = (w.abs().amax() if dim is None else w.abs().amax(dim=dim, keepdim=True)) / qmax
    return torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s


def q_asym(w, bits):
    lo, hi = w.min(), w.max()
    s = (hi - lo) / (2 ** bits - 1)
    z = torch.round(-lo / s)
    return (torch.clamp(torch.round(w / s) + z, 0, 2 ** bits - 1) - z) * s


NF4 = torch.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0])


def q_nf4(w, group=64):
    shp = w.shape
    wg = w.reshape(shp[0], -1, group)
    absmax = wg.abs().amax(-1, keepdim=True)
    x = wg / absmax
    idx = (x.unsqueeze(-1) - NF4).abs().argmin(-1)
    return (NF4[idx] * absmax).reshape(shp)


def rel_err(a, b):
    return ((a - b).norm() / a.norm()).item()


def main() -> None:
    w = make_weight()
    x = torch.randn(64, w.shape[1])
    x[:, :4] *= 30                                          # activation outliers too
    y = x @ w.t()
    rows = [
        ("INT8 sym per-tensor", q_sym(w, 8), 8 + 32 / w.numel()),
        ("INT8 sym per-channel", q_sym(w, 8, dim=1), 8 + 32 / w.shape[1]),
        ("INT4 sym per-tensor", q_sym(w, 4), 4 + 32 / w.numel()),
        ("INT4 asym per-tensor", q_asym(w, 4), 4 + 64 / w.numel()),
        ("INT4 sym per-channel", q_sym(w, 4, dim=1), 4 + 32 / w.shape[1]),
        ("INT4 sym group-128", q_sym(w, 4, group=128), 4 + 16 / 128),
        ("INT4 sym group-32", q_sym(w, 4, group=32), 4 + 16 / 32),
        ("NF4 group-64", q_nf4(w, 64), 4 + 16 / 64),
        ("INT3 sym group-32", q_sym(w, 3, group=32), 3 + 16 / 32),
        ("INT2 sym group-32", q_sym(w, 2, group=32), 2 + 16 / 32),
    ]
    print(f"{'scheme':<26}{'bits/weight':>12}{'weight err':>12}{'output err':>12}")
    for name, wq, bpw in rows:
        print(f"{name:<26}{bpw:>12.3f}{rel_err(w, wq):>12.4f}{rel_err(y, x @ wq.t()):>12.4f}")
    # AWQ-style activation-aware scaling (Lin et al., 2024): protect input channels
    # with large ACTIVATIONS by scaling their weights up before quantization and the
    # activations down at runtime; the product is unchanged, the error is not.
    act = x.abs().mean(0)
    s_awq = (act / act.mean()).clamp(min=1e-3) ** 0.5
    wq_awq = q_sym(w * s_awq, 4, group=128) / s_awq
    print(f"{'INT4 group-128 + AWQ scale':<26}{4 + 16 / 128:>12.3f}{rel_err(w, wq_awq):>12.4f}"
          f"{rel_err(y, x @ wq_awq.t()):>12.4f}")


if __name__ == "__main__":
    main()
