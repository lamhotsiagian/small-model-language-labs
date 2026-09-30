"""
Lab 2, step 2: train a ~10M-parameter decoder from scratch on TinyStories.

Pure PyTorch: the model is slmlab.model.SmallLM (no Hugging Face modelling
code). The script:
  1. loads TinyStories (falls back to a synthetic corpus offline),
  2. tokenizes with a byte-level tokenizer (no tokenizer training needed),
  3. packs documents into fixed windows,
  4. trains with AdamW + WSD schedule (slmlab.train),
  5. samples text with the KV-cached generate() to verify coherence,
  6. writes history.json so calculator.py can compare predicted and
     measured throughput.

Usage:
    python train_tiny.py --preset smoke          # ~1M params, 200 steps, CPU
    python train_tiny.py --preset 10m            # ~9.5M params (Chapter 2 target)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.data import ByteTokenizer, PackedDataset, infinite_loader, load_text_corpus
from slmlab.model import SLMConfig, SmallLM
from slmlab.train import TrainConfig, train

PRESETS = {
    # name: (model kwargs, train kwargs, docs, seq_len)
    "smoke": (dict(d_model=128, n_layers=4, n_heads=4, n_kv_heads=2, d_ff=384),
              dict(steps=200, batch_size=16, lr=3e-3, warmup=20, eval_every=50), 4_000, 128),
    "10m": (dict(d_model=384, n_layers=6, n_heads=6, n_kv_heads=2, d_ff=1024),
            dict(steps=4000, batch_size=32, lr=2e-3, warmup=200, eval_every=250), 200_000, 256),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="smoke")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/tiny")
    args = ap.parse_args()

    mkw, tkw, n_docs, seq = PRESETS[args.preset]
    if args.steps:
        tkw["steps"] = args.steps
    tok = ByteTokenizer()
    docs = load_text_corpus(max_docs=n_docs)
    split = int(0.98 * len(docs))
    train_ds = PackedDataset(docs[:split], tok, seq_len=seq)
    val_ds = PackedDataset(docs[split:], tok, seq_len=seq)
    print(f"[lab02] train tokens={train_ds.n_tokens:,}  val tokens={val_ds.n_tokens:,}")

    cfg = SLMConfig(vocab_size=tok.vocab_size, max_seq_len=seq, **mkw)
    model = SmallLM(cfg)
    print(f"[lab02] params={model.num_params():,} (non-embedding {model.num_params(True):,})")

    tcfg = TrainConfig(out_dir=args.out, **tkw)
    t0 = time.time()
    hist = train(model, infinite_loader(train_ds, tcfg.batch_size),
                 infinite_loader(val_ds, tcfg.batch_size, seed=1), tcfg, device=args.device)
    print(f"[lab02] trained in {time.time() - t0:.0f}s, final val loss {hist[-1]['val_loss']:.3f}"
          f" (bits/byte {hist[-1]['val_loss'] / 0.6931:.3f})")

    # Verify coherent generation: prefill the prompt, then decode with KV cache.
    prompt = torch.tensor([tok.encode("Once upon a time", bos=False)], device=args.device)
    out = model.generate(prompt, max_new_tokens=160, temperature=0.7, top_k=40)
    sample = tok.decode(out[0].tolist())
    print("[lab02] sample:\n" + sample)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "sample.txt").write_text(sample)
    (Path(args.out) / "model_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))


if __name__ == "__main__":
    main()
