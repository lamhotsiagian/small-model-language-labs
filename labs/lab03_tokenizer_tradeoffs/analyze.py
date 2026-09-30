"""
Lab 3, steps 3-4: fertility per language and embedding cost per vocabulary.

Fertility = tokens / whitespace-separated words (lower is better). For code,
words are a poor unit, so we also report bytes per token (higher is better).

Embedding share is computed for a fixed ~300M-parameter backbone: the
non-embedding transformer is held constant and only the vocabulary changes,
so you can see exactly what each extra 16K tokens costs.

The recommendation rule (Chapter 3, section 3.2): choose the smallest vocab
whose fertility on your worst important language is within `--tolerance` of
the best vocab, then check the embedding share is below `--max-embed-share`.

Usage:
    python analyze.py --tokenizers results/tokenizers --data data
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tokenizers import Tokenizer

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.budget import ArchSpec, param_breakdown

# ~300M backbone: 24 layers, d=1024, 16 heads (4 KV), SwiGLU d_ff=2816.
BACKBONE = dict(d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4, d_ff=2816)


def fertility(tok: Tokenizer, text: str) -> dict:
    words = len(text.split())
    n_tok = len(tok.encode(text).ids)
    return {"tokens": n_tok, "fertility": n_tok / max(words, 1),
            "bytes_per_token": len(text.encode("utf-8")) / max(n_tok, 1)}


def vocab_from_name(p: Path) -> int:
    return int(re.search(r"bpe_(\d+)", p.name).group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", default="results/tokenizers")
    ap.add_argument("--data", default="data")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--max-embed-share", type=float, default=0.20)
    ap.add_argument("--tied", action="store_true", default=True)
    a = ap.parse_args()

    held = {f.stem: f.read_text() for f in sorted((Path(a.data) / "heldout").glob("*.txt"))}
    rows = []
    for p in sorted(Path(a.tokenizers).glob("bpe_*.json"), key=vocab_from_name):
        tok = Tokenizer.from_file(str(p))
        v = tok.get_vocab_size()
        spec = ArchSpec(p.stem, vocab=v, tied=a.tied, **BACKBONE)
        b = param_breakdown(spec)
        row = {"vocab": v, "total_params_M": round(b["total"] / 1e6, 1),
               "embed_params_M": round((b["embedding"] + b["lm_head"]) / 1e6, 1),
               "embed_share": round(b["embedding_share"], 3)}
        for lang, text in held.items():
            f = fertility(tok, text)
            row[f"{lang}_fert"] = round(f["fertility"], 3)
            row[f"{lang}_bpt"] = round(f["bytes_per_token"], 2)
        rows.append(row)

    langs = list(held)
    hdr = f"{'vocab':>7} {'params':>8} {'embed':>7} {'share':>6} " + " ".join(f"{l+'_fert':>9}" for l in langs)
    print(hdr)
    for r in rows:
        print(f"{r['vocab']:>7} {r['total_params_M']:>7}M {r['embed_params_M']:>6}M {r['embed_share']:>6.1%} "
              + " ".join(f"{r[l + '_fert']:>9.3f}" for l in langs))

    # Recommendation: smallest vocab within tolerance of the best worst-language fertility.
    worst = [max(r[f"{l}_fert"] for l in langs) for r in rows]
    best = min(worst)
    ok = [r for r, w in zip(rows, worst) if w <= best * (1 + a.tolerance)
          and r["embed_share"] <= a.max_embed_share]
    rec = ok[0] if ok else min(rows, key=lambda r: r["embed_share"])
    print(f"\nrecommended vocab: {rec['vocab']:,} (worst-language fertility within "
          f"{a.tolerance:.0%} of best, embedding share {rec['embed_share']:.1%})")
    Path("results").mkdir(exist_ok=True)
    Path("results/tokenizer_report.json").write_text(json.dumps({"rows": rows, "recommended": rec}, indent=2))


if __name__ == "__main__":
    main()
