"""
Lab 3, step 2: train byte-level BPE tokenizers at several vocabulary sizes.

Design choices (each discussed in Chapter 3):
  * Byte-level BPE with a 256-symbol base alphabet: every string is encodable,
    so there is no <unk> token and no out-of-vocabulary failure mode.
  * Pre-tokenization splits digits into single characters (Llama 3 / Qwen /
    Gemma style), which makes arithmetic tokenization consistent.
  * A small set of special tokens reserved up front: chat-role markers and
    tool-call delimiters. Adding them later means resizing the embeddings.

Usage:
    python train_tokenizers.py --vocab-sizes 16000 32000 64000
    python train_tokenizers.py --data data --vocab-sizes 400 800 1600   # toy corpus
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>",
                  "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]


def build_tokenizer(split_digits: bool = True) -> Tokenizer:
    tok = Tokenizer(models.BPE(byte_fallback=False))
    pre = [pre_tokenizers.Digits(individual_digits=True)] if split_digits else []
    pre.append(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence(pre)
    tok.decoder = decoders.ByteLevel()
    return tok


def train(files, vocab_size: int, split_digits: bool, out: Path) -> Path:
    tok = build_tokenizer(split_digits)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   # all 256 bytes
        show_progress=False,
    )
    tok.train([str(f) for f in files], trainer)
    path = out / f"bpe_{vocab_size}{'' if split_digits else '_nodigitsplit'}.json"
    tok.save(str(path))
    print(f"[lab03] trained {path.name}: actual vocab {tok.get_vocab_size():,}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--vocab-sizes", nargs="+", type=int, default=[16_000, 32_000, 64_000])
    ap.add_argument("--no-digit-split", action="store_true")
    ap.add_argument("--out", default="results/tokenizers")
    a = ap.parse_args()
    files = sorted((Path(a.data) / "train").glob("*.txt"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in a.vocab_sizes:
        train(files, v, not a.no_digit_split, out)


if __name__ == "__main__":
    main()
