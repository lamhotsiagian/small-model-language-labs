"""
Lab 4: Architecture ablations at a fixed parameter and token budget.

Four questions, one controlled experiment each (Chapter 4):

  A. depth vs width     deep-thin  (d=576,  ~30 layers)  vs  wide-shallow (d=1024, ~8 layers)
  B. attention sharing  MHA (kv = heads)  vs  GQA (kv = heads/3)
  C. embedding tying    tied  vs  untied (layers removed to pay for the extra matrix)
  D. (bonus)            QK-norm on/off at a high learning rate (stability)

Fairness rules that make an ablation mean something:
  * every variant gets the SAME number of training tokens, data order, and seed
  * total parameters are matched (within ~3%) by solving for the layer count
  * the LR schedule is identical; only the architecture changes
  * inference throughput is measured with the same prompt, batch, and length

Usage:
    python ablate.py --preset smoke          # ~2M params, CPU, pipeline check only
    python ablate.py --preset 125m           # the chapter experiment (1 GPU, hours)
    python ablate.py --preset 125m --only deep_thin wide_shallow
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.budget import ArchSpec, kv_bytes_per_token, param_breakdown
from slmlab.data import ByteTokenizer, PackedDataset, infinite_loader, load_text_corpus
from slmlab.model import SLMConfig, SmallLM
from slmlab.train import TrainConfig, train


def match_layers(target: int, d: int, h: int, kv: int, dff: int, vocab: int, tied: bool) -> int:
    """Smallest layer count whose total parameters reach the target budget."""
    for L in range(1, 200):
        n = param_breakdown(ArchSpec("x", d, L, h, kv, dff, vocab, tied))["total"]
        if n >= target:
            return L
    raise ValueError("target unreachable")


def variants(target: int, vocab: int, scale: str) -> dict:
    if scale == "smoke":
        thin, wide = dict(d=96, h=3, kv=1, dff=256), dict(d=192, h=6, kv=2, dff=512)
    else:
        thin, wide = dict(d=576, h=9, kv=3, dff=1536), dict(d=1024, h=16, kv=4, dff=2816)
    out = {}
    for name, base, kv_override, tied in [
        ("deep_thin", thin, None, True),          # SmolLM2 / MobileLLM shape
        ("wide_shallow", wide, None, True),
        ("deep_thin_mha", thin, thin["h"], True),  # B: MHA instead of GQA
        ("deep_thin_untied", thin, None, False),   # C: untied embeddings
    ]:
        kv = kv_override or base["kv"]
        L = match_layers(target, base["d"], base["h"], kv, base["dff"], vocab, tied)
        out[name] = SLMConfig(vocab_size=vocab, d_model=base["d"], n_layers=L, n_heads=base["h"],
                              n_kv_heads=kv, d_ff=base["dff"], tie_embeddings=tied)
    out["deep_thin_qknorm"] = SLMConfig(**{**out["deep_thin"].to_dict(), "qk_norm": True})
    return out


@torch.no_grad()
def decode_throughput(model: SmallLM, device: str, prompt_len: int = 128, new: int = 128,
                      batch: int = 8) -> float:
    """Tokens/s for batched KV-cached decoding (same settings for every variant)."""
    model.eval().to(device)
    x = torch.randint(0, model.cfg.vocab_size, (batch, prompt_len), device=device)
    caches = [dict() for _ in model.layers]
    logits, _ = model(x, kv_caches=caches)                       # prefill
    nxt = logits[:, -1:].argmax(-1)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(new):
        logits, _ = model(nxt, kv_caches=caches)
        nxt = logits[:, -1:].argmax(-1)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return batch * new / (time.perf_counter() - t0)


PRESETS = {
    "smoke": dict(target=2_000_000, seq=128, docs=6_000,
                  train=dict(steps=300, batch_size=16, lr=3e-3, warmup=30, eval_every=100)),
    "125m": dict(target=125_000_000, seq=1024, docs=2_000_000,
                 train=dict(steps=20_000, batch_size=32, grad_accum=4, lr=3e-3, warmup=1000,
                            eval_every=1000, eval_batches=50)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="smoke")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    P = PRESETS[a.preset]

    # A byte tokenizer holds tokenization constant across variants. For the full
    # run, swap in the Lab 3 tokenizer (32K) so embedding effects are realistic.
    tok = ByteTokenizer()
    vocab = tok.vocab_size if a.preset == "smoke" else 32_000
    docs = load_text_corpus(max_docs=P["docs"])
    split = int(0.98 * len(docs))
    train_ds = PackedDataset(docs[:split], tok, seq_len=P["seq"])
    val_ds = PackedDataset(docs[split:], tok, seq_len=P["seq"])

    results = {}
    for name, cfg in variants(P["target"], vocab, a.preset).items():
        if a.only and name not in a.only:
            continue
        cfg.max_seq_len = P["seq"]
        torch.manual_seed(0)                                   # same init stream
        model = SmallLM(cfg)
        print(f"\n[lab04] {name}: L={cfg.n_layers} d={cfg.d_model} H={cfg.n_heads} "
              f"KV={cfg.n_kv_heads} tied={cfg.tie_embeddings} params={model.num_params():,}")
        tcfg = TrainConfig(out_dir=f"{a.out}/{name}", seed=0, **P["train"])
        hist = train(model, infinite_loader(train_ds, tcfg.batch_size, seed=0),
                     infinite_loader(val_ds, tcfg.batch_size, seed=1), tcfg, device=a.device)
        spec = ArchSpec(name, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
                        cfg.d_ff, cfg.vocab_size, cfg.tie_embeddings)
        results[name] = {"params": model.num_params(), "layers": cfg.n_layers,
                         "final_val_loss": hist[-1]["val_loss"],
                         "train_tok_per_s": hist[-1]["tok_per_s"],
                         "decode_tok_per_s": round(decode_throughput(model, a.device)),
                         "kv_kb_per_token": kv_bytes_per_token(spec, "bf16") / 1024,
                         "history": hist}

    Path(a.out).mkdir(exist_ok=True)
    Path(f"{a.out}/ablation.json").write_text(json.dumps(results, indent=2))
    print(f"\n{'variant':<18}{'L':>4}{'params':>12}{'val_loss':>10}{'decode t/s':>12}{'KV KB/tok':>11}")
    for n, r in results.items():
        print(f"{n:<18}{r['layers']:>4}{r['params']:>12,}{r['final_val_loss']:>10.4f}"
              f"{r['decode_tok_per_s']:>12}{r['kv_kb_per_token']:>11.2f}")


if __name__ == "__main__":
    main()
