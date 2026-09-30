"""
Lab 3, step 1: assemble a mixed English / Indonesian / code corpus.

Pulls a fixed number of characters per language from public datasets through
`datasets` streaming (no full download), and writes one text file per
language plus a held-out split for fertility measurement. Offline, it falls
back to the tiny `sample_corpus/` shipped with the lab so the pipeline runs.

Mixture matters: a tokenizer learns merges in proportion to what it sees. If
Indonesian is 5% of the training text, Indonesian words will fragment.

Usage:
    python build_corpus.py --chars-per-lang 200_000_000 --mix en=0.5 id=0.3 code=0.2
    python build_corpus.py --offline
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SOURCES = {
    # language: (dataset, config, split, text field)
    "en": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    "id": ("wikimedia/wikipedia", "20231101.id", "train", "text"),
    "code": ("bigcode/the-stack-smol", None, "train", "content"),
}


def stream_chars(name, config, split, field, n_chars):
    from datasets import load_dataset  # type: ignore
    ds = load_dataset(name, config, split=split, streaming=True)
    got, out = 0, []
    for row in ds:
        t = row[field]
        out.append(t)
        got += len(t)
        if got >= n_chars:
            break
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars-per-lang", type=int, default=50_000_000)
    ap.add_argument("--mix", nargs="*", default=["en=0.5", "id=0.3", "code=0.2"])
    ap.add_argument("--heldout-frac", type=float, default=0.02)
    ap.add_argument("--out", default="data")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "heldout").mkdir(parents=True, exist_ok=True)

    if a.offline:
        for f in Path(__file__).with_name("sample_corpus").glob("*.txt"):
            shutil.copy(f, out / "train" / f.name)
            shutil.copy(f, out / "heldout" / f.name)   # toy: same text (demo only)
        print("[lab03] offline sample corpus copied (toy; not for conclusions)")
        return

    mix = {k: float(v) for k, v in (m.split("=") for m in a.mix)}
    total = a.chars_per_lang * len(mix)
    for lang, frac in mix.items():
        n = int(total * frac)
        text = stream_chars(*SOURCES[lang], n_chars=n)
        cut = int(len(text) * (1 - a.heldout_frac))
        (out / "train" / f"{lang}.txt").write_text(text[:cut])
        (out / "heldout" / f"{lang}.txt").write_text(text[cut:])
        print(f"[lab03] {lang}: {n:,} chars ({frac:.0%})")


if __name__ == "__main__":
    main()
