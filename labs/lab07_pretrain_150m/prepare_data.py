"""
Lab 7, step 1: tokenize a pretraining mixture into flat uint16 shards.

Pretraining reads tokens, not text. Tokenizing once, up front, into
memory-mapped binary shards means the training job never waits on a
tokenizer, restarts are instant, and every rank can seek to any offset.

Each shard is a flat array of token IDs with EOS between documents
(sequence packing happens at read time; see pretrain.py). uint16 is enough
for vocabularies below 65,536.

Usage:
    python prepare_data.py --dataset HuggingFaceFW/fineweb-edu --config sample-10BT \
        --tokenizer HuggingFaceTB/SmolLM2-135M --tokens 5e9 --out data/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--tokens", type=float, default=5e9)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    assert len(tok) < 2 ** 16, "uint16 shards need vocab < 65,536"
    eos = tok.eos_token_id
    ds = load_dataset(a.dataset, a.config, split="train", streaming=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    buf, shard, total = [], 0, 0
    target = int(a.tokens) + a.val_tokens

    def flush(name: str) -> None:
        nonlocal buf
        np.array(buf, dtype=np.uint16).tofile(out / name)
        buf = []

    for row in ds:
        buf.extend(tok(row["text"], add_special_tokens=False)["input_ids"] + [eos])
        if total == 0 and len(buf) >= a.val_tokens:      # first chunk -> validation
            flush("val_000.bin"); total = a.val_tokens
        elif len(buf) >= a.shard_tokens:
            total += len(buf)
            flush(f"train_{shard:03d}.bin"); shard += 1
            print(f"[lab07] {total / 1e9:.2f}B tokens")
        if total >= target:
            break
    if buf:
        flush(f"train_{shard:03d}.bin")
    print(f"[lab07] done: {shard + 1} train shards in {out}")


if __name__ == "__main__":
    main()
