"""
Lab 2, step 3: a FLOPs and memory calculator that must match measurement.

For a given SLMConfig the script prints three columns and their ratio:

  analytic   closed-form counts from Chapter 2 (params, 2N / 6N FLOPs, bytes)
  measured   torch.utils.flop_counter.FlopCounterMode on a real forward and
             backward pass, and the real parameter count / tensor bytes
  ratio      measured / analytic (should be within a few percent)

The gap between "2N" and the measured forward FLOPs is the attention term
(QK^T and AV), which grows with sequence length: 4 * L * T * d per token.

Usage:
    python calculator.py                       # the 10M preset
    python calculator.py --d-model 2048 --layers 16 --heads 32 --kv-heads 8 \
        --d-ff 8192 --vocab 128256 --seq 512  # Llama-3.2-1B shaped (params only)
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.flop_counter import FlopCounterMode

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.model import SLMConfig, SmallLM


def analytic(cfg: SLMConfig, seq: int) -> dict:
    d, L, hd = cfg.d_model, cfg.n_layers, cfg.head_dim
    attn = d * cfg.n_heads * hd * 2 + 2 * d * cfg.n_kv_heads * hd       # q, o, k, v
    ffn = 3 * d * cfg.d_ff
    per_layer = attn + ffn + 2 * d
    embed = cfg.vocab_size * d
    n = L * per_layer + embed + d + (0 if cfg.tie_embeddings else embed)
    # Matmul FLOPs: every weight in a linear layer (incl. the LM head, which is
    # a matmul even when tied) costs 2 FLOPs per token. The embedding LOOKUP is free.
    n_matmul = L * (attn + ffn) + embed
    fwd_linear = 2 * n_matmul * seq
    # attention scores: causal => average context ~ seq/2, but kernels compute
    # the full T x T block and mask, so we count full T (what FlopCounter sees).
    fwd_attn = 2 * 2 * L * seq * seq * cfg.n_heads * hd
    return {"params": n, "fwd_flops": fwd_linear + fwd_attn, "fwd_linear": fwd_linear,
            "fwd_attn": fwd_attn, "train_linear": 3 * fwd_linear,
            "weights_bf16_mb": n * 2 / 2 ** 20,
            "adamw_state_mb": n * (4 + 4 + 4) / 2 ** 20,     # fp32 master + m + v
            "kv_cache_per_token_kb": 2 * L * cfg.n_kv_heads * hd * 2 / 1024}


def measured(cfg: SLMConfig, seq: int, batch: int = 1) -> dict:
    model = SmallLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (batch, seq))
    with FlopCounterMode(display=False) as fc:
        _, loss = model(x, x)
    fwd = fc.get_total_flops() / batch
    with FlopCounterMode(display=False) as fc2:
        _, loss = model(x, x)
        loss.backward()
    # NOTE: FlopCounterMode counts matmuls exactly. The fused CPU attention
    # kernel is not registered with it, so "measured" covers the linear layers
    # (the 2N / 6N part) and the attention term is reported analytically.
    return {"params": model.num_params(), "fwd_linear": fwd,
            "train_linear": fc2.get_total_flops() / batch}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=259)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--skip-measure", action="store_true")
    a = ap.parse_args()
    cfg = SLMConfig(vocab_size=a.vocab, d_model=a.d_model, n_layers=a.layers, n_heads=a.heads,
                    n_kv_heads=a.kv_heads, d_ff=a.d_ff, max_seq_len=a.seq)

    an = analytic(cfg, a.seq)
    print(f"config: d={a.d_model} L={a.layers} H={a.heads} KV={a.kv_heads} d_ff={a.d_ff} "
          f"V={a.vocab} T={a.seq}")
    print(f"  weights (bf16)        {an['weights_bf16_mb']:10.1f} MB")
    print(f"  AdamW fp32 states     {an['adamw_state_mb']:10.1f} MB  (master + m + v)")
    print(f"  KV cache / token      {an['kv_cache_per_token_kb']:10.2f} KB (bf16)")
    print(f"  attention share of fwd FLOPs at T={a.seq}: {an['fwd_attn'] / an['fwd_flops']:.1%}")
    if a.skip_measure:
        print(f"  params (analytic)     {an['params']:,}")
        return
    me = measured(cfg, a.seq)
    print(f"\n{'quantity':<22}{'analytic':>16}{'measured':>16}{'ratio':>8}")
    for key in ("params", "fwd_linear", "train_linear"):
        print(f"{key:<22}{an[key]:>16,.0f}{me[key]:>16,.0f}{me[key] / an[key]:>8.3f}")
    print(f"{'fwd_attention':<22}{an['fwd_attn']:>16,.0f}{'(analytic)':>16}")
    print(f"\n2N rule (fwd, per sequence) = {2 * an['params'] * a.seq:,.0f}"
          f"  -> measured linear / 2N = {me['fwd_linear'] / (2 * an['params'] * a.seq):.3f}")
    print(f"backward / forward = {me['train_linear'] / me['fwd_linear'] - 1:.2f}  (the 6N rule assumes 2.00)")


if __name__ == "__main__":
    main()
