"""
slmlab.export
=============

Convert a ``SmallLM`` checkpoint into a Hugging Face ``transformers`` model so
the rest of the ecosystem works on it: lm-evaluation-harness (Labs 7, 19),
TRL (Labs 11-14), vLLM / SGLang (Lab 17), and llama.cpp GGUF conversion
(Labs 16, 18).

Architecture mapping:
  * qk_norm=False -> LlamaForCausalLM
  * qk_norm=True  -> Qwen3ForCausalLM (Llama + per-head RMSNorm on q and k)

RoPE layout: SmallLM rotates interleaved pairs (x0,x1),(x2,x3),... (the
original Meta layout). HF Llama/Qwen rotate halves (x_i, x_{i+hd/2}). The
two are equivalent up to a fixed permutation of each head's q/k output rows,
which we apply here (and to the q/k norm weights, which live in the same
per-head coordinate system).

Usage:
    python -m slmlab.export --ckpt checkpoints/slm150m/step_0004767.pt \
        --config labs/lab07_pretrain_150m/config_150m.yaml \
        --tokenizer HuggingFaceTB/SmolLM2-135M --out exported/slm150m
"""
from __future__ import annotations

import argparse

import torch

from .model import SLMConfig, SmallLM


def _perm(hd: int) -> torch.Tensor:
    return torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)])


def _permute_rows(w: torch.Tensor, n_heads: int, hd: int) -> torch.Tensor:
    return w.view(n_heads, hd, -1)[:, _perm(hd), :].reshape(n_heads * hd, -1)


def to_hf(model: SmallLM):
    from transformers import LlamaConfig, LlamaForCausalLM
    c = model.cfg
    common = dict(vocab_size=c.vocab_size, hidden_size=c.d_model, intermediate_size=c.d_ff,
                  num_hidden_layers=c.n_layers, num_attention_heads=c.n_heads,
                  num_key_value_heads=c.n_kv_heads, max_position_embeddings=c.max_seq_len,
                  rope_theta=c.rope_theta, rms_norm_eps=c.norm_eps,
                  tie_word_embeddings=c.tie_embeddings, head_dim=c.head_dim)
    if c.qk_norm:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        hf = Qwen3ForCausalLM(Qwen3Config(**common, attention_bias=False))
    else:
        hf = LlamaForCausalLM(LlamaConfig(**common, attention_bias=False, mlp_bias=False))
    hd, sd = c.head_dim, {}
    src = model.state_dict()
    sd["model.embed_tokens.weight"] = src["embed.weight"]
    sd["model.norm.weight"] = src["final_norm.weight"]
    sd["lm_head.weight"] = src["lm_head.weight"]
    for i in range(c.n_layers):
        p, q = f"layers.{i}.", f"model.layers.{i}."
        sd[q + "input_layernorm.weight"] = src[p + "attn_norm.weight"]
        sd[q + "post_attention_layernorm.weight"] = src[p + "ffn_norm.weight"]
        sd[q + "self_attn.q_proj.weight"] = _permute_rows(src[p + "attn.q_proj.weight"], c.n_heads, hd)
        sd[q + "self_attn.k_proj.weight"] = _permute_rows(src[p + "attn.k_proj.weight"], c.n_kv_heads, hd)
        sd[q + "self_attn.v_proj.weight"] = src[p + "attn.v_proj.weight"]
        sd[q + "self_attn.o_proj.weight"] = src[p + "attn.o_proj.weight"]
        if c.qk_norm:
            sd[q + "self_attn.q_norm.weight"] = src[p + "attn.q_norm.weight"][_perm(hd)]
            sd[q + "self_attn.k_norm.weight"] = src[p + "attn.k_norm.weight"][_perm(hd)]
        sd[q + "mlp.gate_proj.weight"] = src[p + "ffn.gate.weight"]
        sd[q + "mlp.up_proj.weight"] = src[p + "ffn.up.weight"]
        sd[q + "mlp.down_proj.weight"] = src[p + "ffn.down.weight"]
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    return hf


def main() -> None:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = SLMConfig(**yaml.safe_load(open(a.config))["model"])
    model = SmallLM(cfg)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    hf = to_hf(model)
    hf.save_pretrained(a.out, safe_serialization=True)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(a.tokenizer).save_pretrained(a.out)
    print(f"[export] wrote {a.out} ({type(hf).__name__})")


if __name__ == "__main__":
    main()
