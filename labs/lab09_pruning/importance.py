"""
Lab 9, step 1: importance scores for depth and width pruning.

Works on any Hugging Face Llama-family model (Llama 3.x, Qwen2.5/3, SmolLM).

Depth (whole layers):
  block_influence   1 - cos(h_in, h_out) per layer, averaged over tokens
                    (ShortGPT's "Block Influence"; Men et al., 2024). Layers
                    that barely rotate the residual stream are removable.

Width (Minitron-style activation importance; Muralidharan et al., 2024):
  ffn_neuron        mean |SiLU(gate) * up| per intermediate channel
  attn_head         mean L2 norm of each head's output (before o_proj)
  (embedding channels can be scored from RMSNorm outputs the same way)

Unstructured (for comparison, Section 9.2):
  wanda             |W_ij| * ||X_j||_2  (Sun et al., 2024)

All scores come from forward hooks over a small calibration set (a few
hundred sequences is enough; importance ranks stabilise quickly).

Usage:
    python importance.py --model meta-llama/Llama-3.2-3B --calib 256 --out results/importance.pt
    python importance.py --selftest
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_importance(model, batches) -> dict:
    layers = model.model.layers
    cfg = model.config
    n_heads, hd = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or hd
    acc = defaultdict(float)
    ffn = [torch.zeros(cfg.intermediate_size) for _ in layers]
    heads = [torch.zeros(n_heads) for _ in layers]
    bi = torch.zeros(len(layers))
    hooks, n_tok = [], 0

    def ffn_hook(i):
        def h(mod, inp, out):
            # input to down_proj is SiLU(gate) * up: the per-channel activation
            ffn[i] += inp[0].abs().float().sum(dim=(0, 1)).cpu()
        return h

    def head_hook(i):
        def h(mod, inp, out):
            x = inp[0].float()                                      # [B, T, H*hd] before o_proj
            heads[i] += x.view(*x.shape[:2], n_heads, hd).norm(dim=-1).sum(dim=(0, 1)).cpu()
        return h

    def block_hook(i):
        def h(mod, inp, out):
            h_in = inp[0].float()
            h_out = (out[0] if isinstance(out, tuple) else out).float()
            bi[i] += (1 - F.cosine_similarity(h_in, h_out, dim=-1)).sum().cpu()
        return h

    for i, layer in enumerate(layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(ffn_hook(i)))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(head_hook(i)))
        hooks.append(layer.register_forward_hook(block_hook(i)))
    for ids in batches:
        model(ids.to(model.device))
        n_tok += ids.numel()
    for h in hooks:
        h.remove()
    return {"block_influence": bi / n_tok, "ffn": [f / n_tok for f in ffn],
            "heads": [h / n_tok for h in heads], "n_tokens": n_tok}


def wanda_scores(weight: torch.Tensor, act_norm: torch.Tensor) -> torch.Tensor:
    """|W| * ||X|| broadcast over output rows; prune the lowest per output row."""
    return weight.abs() * act_norm[None, :]


def tiny_llama(layers=6, d=128, ff=384, heads=4, kv=2, vocab=512):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=vocab, hidden_size=d, intermediate_size=ff,
                                        num_hidden_layers=layers, num_attention_heads=heads,
                                        num_key_value_heads=kv, max_position_embeddings=256)).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--out", default="results/importance.pt")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        model = tiny_llama()
        batches = [torch.randint(0, 512, (4, 64)) for _ in range(4)]
        imp = compute_importance(model, batches)
        print("block influence per layer:", [round(v, 4) for v in imp["block_influence"].tolist()])
        print("layer 0 top-5 FFN channels:", imp["ffn"][0].topk(5).indices.tolist())
        print("layer 0 head importance:", [round(v, 3) for v in imp["heads"][0].tolist()])
        return
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    batches = []
    for row in ds:
        ids = tok(row["text"], return_tensors="pt", truncation=True, max_length=a.seq).input_ids
        if ids.shape[1] >= a.seq // 2:
            batches.append(ids)
        if len(batches) >= a.calib:
            break
    torch.save(compute_importance(model, batches), a.out)
    print(f"[lab09] saved {a.out}")


if __name__ == "__main__":
    main()
