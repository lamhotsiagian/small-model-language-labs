"""
Lab 10, steps 2-3: extend an 8K model to 32K with YaRN + short continued pretraining.

Recipe (Peng et al., 2024; Chapter 10, sections 10.1 and 10.4):
  1. set rope_scaling = {"rope_type": "yarn", "factor": 4, "original_max_position_embeddings": 8192}
     and max_position_embeddings = 32768 in the config (no weights change)
  2. continue pretraining on LONG documents packed to 32K with document
     masking, mixed with a share of short data so short-context quality holds
  3. use a low LR (about 1/10 of the pretraining peak) for a few hundred
     million to ~1B tokens

Default model: HuggingFaceTB/SmolLM2-1.7B (8,192-token native context).

Usage:
    python extend.py --model HuggingFaceTB/SmolLM2-1.7B --factor 4 --tokens 5e8 \
        --long-frac 0.7 --out checkpoints/smollm2_1p7b_32k
    python extend.py --model HuggingFaceTB/SmolLM2-1.7B --factor 4 --no-train   # zero-shot YaRN
"""
from __future__ import annotations

import argparse
import json
import random

import torch


def long_docs(tok, min_tokens: int):
    """Stream documents that are genuinely long (books, long web pages, code files)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    for row in ds:
        ids = tok(row["text"], add_special_tokens=False).input_ids
        if len(ids) >= min_tokens:
            yield ids


def short_docs(tok):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    for row in ds:
        yield tok(row["text"], add_special_tokens=False).input_ids


def packed(tok, seq: int, long_frac: float, seed: int = 0):
    """Yield (input_ids, position_ids) of length seq. position_ids restart at
    every document boundary; with FlashAttention-2 in transformers this gives
    document masking (no attention across packed documents)."""
    rng = random.Random(seed)
    L, S = long_docs(tok, seq // 4), short_docs(tok)
    ids, pos = [], []
    while True:
        doc = next(L) if rng.random() < long_frac else next(S)
        doc = doc[:seq] + [tok.eos_token_id]
        ids += doc
        pos += list(range(len(doc)))
        if len(ids) >= seq:
            yield torch.tensor(ids[:seq]), torch.tensor(pos[:seq])
            ids, pos = [], []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("--tokens", type=float, default=5e8)
    ap.add_argument("--long-frac", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--out", default="checkpoints/smollm2_1p7b_32k")
    a = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(a.model)
    orig = cfg.max_position_embeddings
    cfg.rope_scaling = {"rope_type": "yarn", "factor": a.factor,
                        "original_max_position_embeddings": orig}
    cfg.max_position_embeddings = int(orig * a.factor)
    print(f"[lab10] {a.model}: {orig} -> {cfg.max_position_embeddings} tokens (YaRN x{a.factor:g})")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").cuda()
    if not a.no_train:
        model.gradient_checkpointing_enable()
        seq = cfg.max_position_embeddings
        steps = int(a.tokens // seq)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
        model.train()
        for step, (ids, pos) in enumerate(packed(tok, seq, a.long_frac)):
            if step >= steps:
                break
            lr = a.lr * min(1.0, (step + 1) / 20)
            for g in opt.param_groups:
                g["lr"] = lr
            ids, pos = ids[None].cuda(), pos[None].cuda()
            loss = model(input_ids=ids, position_ids=pos, labels=ids).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % 20 == 0:
                print(json.dumps({"step": step, "of": steps, "loss": round(loss.item(), 4),
                                  "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}))
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"[lab10] saved {a.out}")


if __name__ == "__main__":
    main()
