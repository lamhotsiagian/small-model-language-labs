"""
Lab 9, step 2: structured pruning of a Llama-family model (depth + width).

Given importance scores from importance.py, produce a smaller DENSE model:

  depth   drop the k layers with the lowest block influence (never the first
          or last layer, which consistently score as essential)
  width   keep the top-m FFN channels per layer: slice rows of gate_proj and
          up_proj and the matching columns of down_proj
  heads   keep the top KV groups per layer: slice q/k/v rows and o_proj
          columns group-wise, so GQA structure is preserved

The result is a standard checkpoint (smaller config.json, same code path),
which is why structured pruning gives real speedups on every runtime while
unstructured sparsity needs special kernels.

Usage:
    python prune.py --model meta-llama/Llama-3.2-3B --importance results/importance.pt \
        --drop-layers 10 --ffn-keep 0.5 --kv-keep 1.0 --out checkpoints/pruned_1p5b
    python prune.py --selftest
"""
from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn


def _slice_linear(lin: nn.Linear, idx: torch.Tensor, dim: int) -> nn.Linear:
    """Keep `idx` along output rows (dim=0) or input columns (dim=1)."""
    w = lin.weight.data.index_select(dim, idx.to(lin.weight.device))
    out_f, in_f = (len(idx), lin.in_features) if dim == 0 else (lin.out_features, len(idx))
    new = nn.Linear(in_f, out_f, bias=lin.bias is not None, dtype=w.dtype, device=w.device)
    new.weight.data.copy_(w)
    if lin.bias is not None:
        new.bias.data.copy_(lin.bias.data if dim == 1 else lin.bias.data[idx])
    return new


def prune_ffn(layer, scores: torch.Tensor, keep: int) -> None:
    idx = scores.topk(keep).indices.sort().values
    mlp = layer.mlp
    mlp.gate_proj = _slice_linear(mlp.gate_proj, idx, 0)
    mlp.up_proj = _slice_linear(mlp.up_proj, idx, 0)
    mlp.down_proj = _slice_linear(mlp.down_proj, idx, 1)


def prune_kv_groups(layer, head_scores: torch.Tensor, n_heads: int, n_kv: int, hd: int,
                    keep_groups: int) -> None:
    """Score each KV group by the summed importance of its query heads."""
    per_group = n_heads // n_kv
    g_scores = head_scores.view(n_kv, per_group).sum(-1)
    groups = g_scores.topk(keep_groups).indices.sort().values
    q_idx = torch.cat([torch.arange(g * per_group * hd, (g + 1) * per_group * hd) for g in groups])
    kv_idx = torch.cat([torch.arange(g * hd, (g + 1) * hd) for g in groups])
    at = layer.self_attn
    at.q_proj = _slice_linear(at.q_proj, q_idx, 0)
    at.k_proj = _slice_linear(at.k_proj, kv_idx, 0)
    at.v_proj = _slice_linear(at.v_proj, kv_idx, 0)
    at.o_proj = _slice_linear(at.o_proj, q_idx, 1)


def prune_model(model, imp: dict, drop_layers: int, ffn_keep: float, kv_keep: float):
    cfg = model.config
    L = cfg.num_hidden_layers
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    # --- depth: protect first and last layer ---
    bi = imp["block_influence"].clone()
    bi[0] = bi[-1] = float("inf")
    drop = set(bi.topk(drop_layers, largest=False).indices.tolist()) if drop_layers else set()
    keep_layers = [i for i in range(L) if i not in drop]
    # --- width ---
    ff_keep = int(round(cfg.intermediate_size * ffn_keep / 64) * 64)       # keep kernel-friendly sizes
    kv_groups = max(1, int(round(cfg.num_key_value_heads * kv_keep)))
    new_layers = nn.ModuleList()
    for i in keep_layers:
        layer = model.model.layers[i]
        if ff_keep < cfg.intermediate_size:
            prune_ffn(layer, imp["ffn"][i], ff_keep)
        if kv_groups < cfg.num_key_value_heads:
            prune_kv_groups(layer, imp["heads"][i], cfg.num_attention_heads,
                            cfg.num_key_value_heads, hd, kv_groups)
        new_layers.append(layer)
    for j, layer in enumerate(new_layers):          # keep layer_idx consistent for KV caches
        layer.self_attn.layer_idx = j
    model.model.layers = new_layers
    per_group = cfg.num_attention_heads // cfg.num_key_value_heads
    cfg.num_hidden_layers = len(new_layers)
    cfg.intermediate_size = ff_keep
    cfg.num_key_value_heads = kv_groups
    cfg.num_attention_heads = kv_groups * per_group
    cfg.head_dim = hd
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = [cfg.layer_types[i] for i in keep_layers]
    return model, sorted(drop)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--importance")
    ap.add_argument("--drop-layers", type=int, default=10)
    ap.add_argument("--ffn-keep", type=float, default=0.5)
    ap.add_argument("--kv-keep", type=float, default=1.0)
    ap.add_argument("--out", default="checkpoints/pruned")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        from importance import compute_importance, tiny_llama
        model = tiny_llama()
        before = sum(p.numel() for p in model.parameters())
        imp = compute_importance(model, [torch.randint(0, 512, (4, 64)) for _ in range(4)])
        pruned, dropped = prune_model(copy.deepcopy(model), imp, drop_layers=2, ffn_keep=0.5, kv_keep=0.5)
        after = sum(p.numel() for p in pruned.parameters())
        x = torch.randint(0, 512, (2, 32))
        out = pruned(x).logits
        # generate() exercises the KV cache with re-indexed layers
        gen = pruned.generate(x[:, :8], max_new_tokens=8, do_sample=False)
        print(f"dropped layers {dropped}; params {before:,} -> {after:,} ({after / before:.1%}); "
              f"logits {tuple(out.shape)}; generate ok {tuple(gen.shape)}")
        print(pruned.config.num_hidden_layers, pruned.config.intermediate_size,
              pruned.config.num_attention_heads, pruned.config.num_key_value_heads)
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    before = sum(p.numel() for p in model.parameters())
    model, dropped = prune_model(model, torch.load(a.importance), a.drop_layers, a.ffn_keep, a.kv_keep)
    after = sum(p.numel() for p in model.parameters())
    model.save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
    print(f"[lab09] dropped {dropped}; {before / 1e9:.2f}B -> {after / 1e9:.2f}B params; saved {a.out}")


if __name__ == "__main__":
    main()
