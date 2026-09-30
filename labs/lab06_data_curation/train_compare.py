"""
Lab 6, step 7: does curation pay? Train two identical 60M models, one on raw
web text and one on the curated mixture, with the SAME token budget, and
compare them on:

  * validation loss on a neutral held-out set (e.g. a Wikipedia sample), not
    on either training distribution, so neither model gets a home advantage
  * a downstream probe via lm-evaluation-harness (HellaSwag, ARC-Easy)

Usage:
    python train_compare.py --raw data/raw.jsonl --curated data/final.jsonl \
        --heldout data/heldout.jsonl --tokenizer ../lab03_tokenizer_tradeoffs/results/tokenizers/bpe_32000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.data import PackedDataset, infinite_loader
from slmlab.model import SLMConfig, SmallLM
from slmlab.train import TrainConfig, evaluate, train


class HFTok:
    """Adapter so a `tokenizers` JSON file plugs into PackedDataset."""

    def __init__(self, path: str) -> None:
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(path)
        self.eos_token_id = self.tok.token_to_id("<|endoftext|>")
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, text: str):
        return self.tok.encode(text).ids


def load(path: str, n: int | None = None):
    rows = [json.loads(l)["text"] for l in open(path)]
    return rows[:n] if n else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--curated", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tokens", type=float, default=1.2e9)   # ~20 tokens/param for 60M
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tok = HFTok(a.tokenizer)
    seq, batch = 1024, 32
    steps = int(a.tokens / (seq * batch))
    held = PackedDataset(load(a.heldout), tok, seq)
    results = {}
    for name, path in [("raw", a.raw), ("curated", a.curated)]:
        ds = PackedDataset(load(path), tok, seq, max_tokens=int(a.tokens) + seq)
        cfg = SLMConfig(vocab_size=tok.vocab_size, d_model=512, n_layers=16, n_heads=8,
                        n_kv_heads=2, max_seq_len=seq)                      # ~60M with 32K vocab
        torch.manual_seed(0)
        model = SmallLM(cfg)
        tcfg = TrainConfig(steps=steps, batch_size=batch, lr=2e-3, warmup=500,
                           eval_every=max(1, steps // 10), out_dir=f"results/{name}")
        train(model, infinite_loader(ds, batch), infinite_loader(held, batch, seed=1),
              tcfg, device=a.device)
        results[name] = {"params": model.num_params(),
                         "heldout_loss": evaluate(model, infinite_loader(held, batch, seed=2),
                                                  100, a.device)}
        print(name, results[name])
    Path("results/curation_impact.json").write_text(json.dumps(results, indent=2))
    d = results["raw"]["heldout_loss"] - results["curated"]["heldout_loss"]
    print(f"[lab06] curated model is {d:+.4f} nats/token better on the neutral held-out set")
    print("next: lm_eval --model hf --model_args pretrained=<exported ckpt> --tasks hellaswag,arc_easy")


if __name__ == "__main__":
    main()
