"""
slmlab.model
============

A compact, dependency-free (pure PyTorch) implementation of the decoder-only,
pre-norm transformer that nearly every modern small language model uses:

    tokens -> embedding (tied) -> N x [RMSNorm -> GQA+RoPE -> +res ->
                                        RMSNorm -> SwiGLU  -> +res]
           -> final RMSNorm -> LM head (shared with embedding) -> logits

Every architectural knob discussed in the book is a field on ``SLMConfig`` so
the ablation, scaling-law, pruning, and long-context labs can reuse one model:

* ``n_kv_heads``      MHA (== n_heads), GQA (divides n_heads), MQA (== 1)
* ``tie_embeddings``  share input embedding and output projection
* ``qk_norm``         RMSNorm on queries and keys (Qwen3 / Gemma 3 style)
* ``sliding_window``  local attention window for "local" layers
* ``global_every``    every k-th layer is global, the rest are local
* ``logit_softcap``   tanh soft-capping of final logits (Gemma 2 style)
* ``rope_theta`` / ``rope_scaling`` for long-context experiments (Lab 10)

The code favours readability over speed, but it uses
``torch.nn.functional.scaled_dot_product_attention`` so it runs with fused
kernels (FlashAttention / memory-efficient attention) when they are available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SLMConfig:
    vocab_size: int = 32_000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2              # GQA ratio = n_heads / n_kv_heads
    d_ff: Optional[int] = None       # defaults to ~8/3 * d_model rounded to 64
    max_seq_len: int = 1024
    rope_theta: float = 10_000.0
    rope_scaling: Optional[dict] = None   # {"type": "linear"|"ntk"|"yarn", "factor": s, "orig_max": L}
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    qk_norm: bool = False
    sliding_window: Optional[int] = None  # None = every layer is global
    global_every: int = 1                 # with sliding_window: 1 global per k layers
    logit_softcap: Optional[float] = None
    dropout: float = 0.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.d_ff is None:
            # SwiGLU keeps FLOPs comparable to a 4x GELU FFN with 8/3 * d.
            self.d_ff = int(math.ceil((8 / 3) * self.d_model / 64) * 64)
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMSNorm: rescale by root-mean-square, no mean-centering, no bias.

    Cheaper than LayerNorm (one reduction instead of two) and, in pre-norm
    position, the de-facto standard for SLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for stability, then cast back (matters in bf16/fp16).
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def rope_frequencies(cfg: SLMConfig, seq_len: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) tables of shape [seq_len, head_dim/2].

    Supports three context-extension schemes used in Chapter 10:
      * linear  (Position Interpolation): positions are divided by ``factor``
      * ntk     (NTK-aware): the base theta is raised so low frequencies stretch
      * yarn    (simplified): NTK-by-parts ramp between interpolated and
                original frequencies plus an attention temperature (applied in
                attention via ``yarn_mscale``)
    """
    hd = cfg.head_dim
    theta = cfg.rope_theta
    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    pos = torch.arange(seq_len, device=device).float()

    sc = cfg.rope_scaling or {}
    kind, factor = sc.get("type"), float(sc.get("factor", 1.0))
    if kind == "linear":
        pos = pos / factor
    elif kind == "ntk":
        theta_ntk = theta * factor ** (hd / (hd - 2))
        inv_freq = 1.0 / (theta_ntk ** (torch.arange(0, hd, 2, device=device).float() / hd))
    elif kind == "yarn":
        orig = float(sc.get("orig_max", cfg.max_seq_len))
        beta_fast, beta_slow = float(sc.get("beta_fast", 32)), float(sc.get("beta_slow", 1))
        # wavelength of each dimension pair, measured in "rotations per context"
        rotations = orig * inv_freq / (2 * math.pi)
        # ramp = 1 -> keep original freq (high freq), 0 -> fully interpolate
        ramp = ((rotations - beta_slow) / (beta_fast - beta_slow)).clamp(0, 1)
        inv_freq = inv_freq / factor * (1 - ramp) + inv_freq * ramp

    angles = torch.outer(pos, inv_freq)
    return angles.cos(), angles.sin()


def yarn_mscale(cfg: SLMConfig) -> float:
    sc = cfg.rope_scaling or {}
    if sc.get("type") != "yarn":
        return 1.0
    return 0.1 * math.log(float(sc.get("factor", 1.0))) + 1.0


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x_even, x_odd) by position-dependent angles.

    x: [batch, heads, seq, head_dim]; cos/sin: [seq, head_dim/2]
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class GroupedQueryAttention(nn.Module):
    """Causal self-attention with a configurable number of KV heads.

    n_kv_heads == n_heads -> MHA; 1 -> MQA; anything in between -> GQA.
    The KV cache shrinks by n_heads / n_kv_heads, which is why GQA is the
    single most important inference-memory lever for SLMs.
    """

    def __init__(self, cfg: SLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else None
        # Local/global interleaving (Gemma 2/3 style). Layer (k-1) of every k
        # layers is global; the others use a sliding window.
        self.window = None
        if cfg.sliding_window and (layer_idx + 1) % cfg.global_every != 0:
            self.window = cfg.sliding_window
        self.mscale = yarn_mscale(cfg)

    def forward(self, x, cos, sin, kv_cache: Optional[dict] = None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)

        # Positions for this chunk start after anything already cached.
        start = 0 if kv_cache is None or "k" not in kv_cache else kv_cache["k"].shape[2]
        q = apply_rope(q, cos[start:start + T], sin[start:start + T])
        k = apply_rope(k, cos[start:start + T], sin[start:start + T])

        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v

        # Expand KV heads to match query heads (the cache itself stays small).
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        S = k.shape[2]
        # Build a causal (optionally banded) mask aligned to the query offset.
        q_pos = torch.arange(start, start + T, device=x.device)[:, None]
        k_pos = torch.arange(S, device=x.device)[None, :]
        mask = k_pos <= q_pos
        if self.window is not None:
            mask = mask & (k_pos > q_pos - self.window)

        scale = self.mscale / math.sqrt(self.hd)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale,
                                           dropout_p=self.cfg.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.hd)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    """FFN(x) = W_down( SiLU(W_gate x) * (W_up x) ). Three matrices, no bias."""

    def __init__(self, cfg: SLMConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: SLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        # Pre-norm residual stream: each sub-layer READS a normalised copy and
        # WRITES an additive update. The stream itself is never normalised.
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class SmallLM(nn.Module):
    def __init__(self, cfg: SLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(DecoderBlock(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # Saves vocab x d_model parameters: 20-30% of a sub-1B model.
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)
        # Scaled init for residual output projections (GPT-2 / Llama practice).
        for n, p in self.named_parameters():
            if n.endswith("o_proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))
        self._rope_cache: dict = {}

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def rope(self, seq_len: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, str(device))
        if key not in self._rope_cache:
            self._rope_cache[key] = rope_frequencies(self.cfg, seq_len, device)
        return self._rope_cache[key]

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                kv_caches: Optional[list] = None, z_loss: float = 0.0):
        B, T = idx.shape
        past = 0 if not kv_caches or "k" not in kv_caches[0] else kv_caches[0]["k"].shape[2]
        cos, sin = self.rope(max(self.cfg.max_seq_len, past + T), idx.device)
        x = self.embed(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, None if kv_caches is None else kv_caches[i])
        logits = self.lm_head(self.final_norm(x))
        if self.cfg.logit_softcap:
            c = self.cfg.logit_softcap
            logits = c * torch.tanh(logits / c)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-100)
        if z_loss > 0:
            # PaLM-style z-loss keeps the softmax normaliser near 1 (log Z ~ 0),
            # which suppresses the logit drift behind many loss spikes.
            log_z = torch.logsumexp(logits.float(), dim=-1)
            loss = loss + z_loss * (log_z ** 2).mean()
        return logits, loss

    # -------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 64, temperature: float = 0.8,
                 top_k: Optional[int] = 50) -> torch.Tensor:
        """Incremental decoding with a per-layer KV cache (prefill, then decode)."""
        self.eval()
        caches = [dict() for _ in self.layers]
        logits, _ = self(idx, kv_caches=caches)            # prefill: whole prompt at once
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            logits, _ = self(nxt, kv_caches=caches)          # decode: one token per step
        return idx

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())    # tied weights counted once
        if non_embedding:
            n -= self.embed.weight.numel()
        return n
