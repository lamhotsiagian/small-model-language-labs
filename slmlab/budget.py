"""
slmlab.budget
=============

Closed-form accounting for small language models: parameters, FLOPs, KV cache,
and the device-memory plan. Every number printed in Chapters 1, 2, 4, 16, and
18 comes from these functions, so they are written to be read.

All formulas assume the decoder-only, pre-norm, SwiGLU, GQA architecture
described in Part B of the outline (bias-free linear layers, RMSNorm weights).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5,
         "q4_k_m": 4.85 / 8, "q5_k_m": 5.69 / 8, "q8_0": 8.5 / 8, "q3_k_m": 3.91 / 8,
         "q2_k": 3.35 / 8}


@dataclass(frozen=True)
class ArchSpec:
    """The handful of numbers that fully determine an SLM's budget."""
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab: int
    tied: bool = True
    head_dim: Optional[int] = None   # some models decouple head_dim from d_model/n_heads

    @property
    def hd(self) -> int:
        return self.head_dim or self.d_model // self.n_heads


# Published configurations (config.json on the Hugging Face Hub, 2024-2025).
ZOO = {
    "SmolLM2-135M":   ArchSpec("SmolLM2-135M",   576, 30,  9, 3, 1536,  49_152),
    "SmolLM2-360M":   ArchSpec("SmolLM2-360M",   960, 32, 15, 5, 2560,  49_152),
    "Qwen2.5-0.5B":   ArchSpec("Qwen2.5-0.5B",   896, 24, 14, 2, 4864, 151_936),
    "Llama-3.2-1B":   ArchSpec("Llama-3.2-1B",  2048, 16, 32, 8, 8192, 128_256),
    "Qwen2.5-1.5B":   ArchSpec("Qwen2.5-1.5B",  1536, 28, 12, 2, 8960, 151_936),
    "SmolLM2-1.7B":   ArchSpec("SmolLM2-1.7B",  2048, 24, 32, 32, 8192, 49_152),
    "Llama-3.2-3B":   ArchSpec("Llama-3.2-3B",  3072, 28, 24, 8, 8192, 128_256),
    "Phi-3-mini":     ArchSpec("Phi-3-mini",    3072, 32, 32, 32, 8192, 32_064, tied=False),
}


def attention_params(a: ArchSpec) -> int:
    q = a.d_model * a.n_heads * a.hd
    kv = 2 * a.d_model * a.n_kv_heads * a.hd
    o = a.n_heads * a.hd * a.d_model
    return q + kv + o


def ffn_params(a: ArchSpec) -> int:
    return 3 * a.d_model * a.d_ff            # gate, up, down


def param_breakdown(a: ArchSpec) -> dict:
    per_layer = attention_params(a) + ffn_params(a) + 2 * a.d_model   # + 2 RMSNorm weights
    embed = a.vocab * a.d_model
    head = 0 if a.tied else embed
    total = a.n_layers * per_layer + embed + head + a.d_model          # + final norm
    return {
        "attention_per_layer": attention_params(a),
        "ffn_per_layer": ffn_params(a),
        "per_layer": per_layer,
        "all_layers": a.n_layers * per_layer,
        "embedding": embed,
        "lm_head": head,
        "total": total,
        "embedding_share": (embed + head) / total,
        "ffn_share_of_layer": ffn_params(a) / per_layer,
    }


def kv_bytes_per_token(a: ArchSpec, dtype: str = "fp16") -> float:
    """2 (K and V) x layers x kv_heads x head_dim x bytes."""
    return 2 * a.n_layers * a.n_kv_heads * a.hd * BYTES[dtype]


def flops_per_token(n_params: int, training: bool = False) -> float:
    """~2N per token forward, ~6N forward+backward (Kaplan et al., 2020)."""
    return (6 if training else 2) * n_params


def attention_flops_per_token(a: ArchSpec, context: int) -> float:
    """Extra forward FLOPs from QK^T and AV, which grow with context length."""
    return 2 * 2 * a.n_layers * context * a.n_heads * a.hd


def decode_tokens_per_sec_bound(weight_bytes: float, bandwidth_gbs: float) -> float:
    """Memory-bandwidth roofline for batch-1 decode: every weight is read once
    per generated token, so tokens/s <= bandwidth / model bytes."""
    return bandwidth_gbs * 1e9 / weight_bytes


def memory_plan(a: ArchSpec, weight_dtype: str = "q4_k_m", kv_dtype: str = "fp16",
                context: int = 4096, batch: int = 1, runtime_overhead_mb: float = 300) -> dict:
    """What actually has to fit in device RAM when you ship an SLM."""
    n = param_breakdown(a)["total"]
    w = n * BYTES[weight_dtype]
    kv = kv_bytes_per_token(a, kv_dtype) * context * batch
    # Activations for one decode step are tiny; prefill dominates. A chunked
    # prefill of 512 tokens with d_ff-wide intermediates is a safe estimate.
    act = 512 * batch * (a.d_model * 4 + a.d_ff * 2) * 2
    total = w + kv + act + runtime_overhead_mb * 2 ** 20
    return {"weights_mb": w / 2 ** 20, "kv_mb": kv / 2 ** 20, "activations_mb": act / 2 ** 20,
            "overhead_mb": runtime_overhead_mb, "total_mb": total / 2 ** 20}


def fmt(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


if __name__ == "__main__":
    print(f"{'model':<14}{'total':>9}{'embed%':>8}{'FFN%L':>7}{'KV/tok':>9}{'KV@8K':>9}")
    for spec in ZOO.values():
        b = param_breakdown(spec)
        kv = kv_bytes_per_token(spec)
        print(f"{spec.name:<14}{fmt(b['total']):>9}{b['embedding_share']*100:>7.1f}%"
              f"{b['ffn_share_of_layer']*100:>6.1f}%{kv/1024:>7.0f}KB{kv*8192/2**20:>7.0f}MB")
