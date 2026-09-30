"""
slmlab.hfutils
==============

Thin helpers around Hugging Face ``transformers`` used by the labs that work
with pretrained checkpoints (1, 8, 9, 11-20). Kept deliberately small: the
point of the labs is the system behaviour, not a framework.
"""
from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, dtype: str = "auto", device: Optional[str] = None,
               revision: Optional[str] = None):
    """Load (tokenizer, model). Pin ``revision`` to a commit SHA in production."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or best_device()
    torch_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype,
                                                 revision=revision).to(device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


def free(model=None) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def peak_memory():
    """Yield a dict that receives peak GPU memory (MB) for the enclosed block."""
    out = {"peak_mb": None}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    yield out
    if torch.cuda.is_available():
        out["peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    new_tokens: int
    ttft_s: float
    total_s: float

    @property
    def tpot_s(self) -> float:
        """Time per output token after the first (decode speed)."""
        return (self.total_s - self.ttft_s) / max(1, self.new_tokens - 1)


@torch.no_grad()
def chat_generate(tok, model, messages, max_new_tokens: int = 256,
                  temperature: float = 0.0, **template_kwargs) -> GenResult:
    """Apply the model's chat template, generate, and time prefill vs decode.

    TTFT is measured with a 1-token generate call (prefill + one step). It is
    an approximation of what a streaming server reports, adequate for ranking.
    """
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                     **template_kwargs)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    gen_kw = dict(do_sample=temperature > 0, pad_token_id=tok.pad_token_id)
    if temperature > 0:
        gen_kw["temperature"] = temperature

    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    sync(); t0 = time.perf_counter()
    model.generate(**enc, max_new_tokens=1, **gen_kw)
    sync(); ttft = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=max_new_tokens, **gen_kw)
    sync(); total = time.perf_counter() - t0
    new = out[0, enc["input_ids"].shape[1]:]
    return GenResult(tok.decode(new, skip_special_tokens=True), enc["input_ids"].shape[1],
                     len(new), ttft, total)


@torch.no_grad()
def sequence_logprob(tok, model, prompt: str, completion: str) -> float:
    """Sum of log p(completion | prompt): the primitive behind DPO, KD, and eval."""
    p = tok(prompt, return_tensors="pt").input_ids
    c = tok(completion, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([p, c], dim=1).to(model.device)
    logits = model(ids).logits[:, :-1].float()
    logp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return logp[:, p.shape[1] - 1:].sum().item()
