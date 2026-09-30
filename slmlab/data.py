"""
slmlab.data
===========

Data plumbing shared by the pretraining-style labs (2, 4, 5, 7, 10).

* ``load_text_corpus``  pulls TinyStories (or any HF dataset with a text field)
  when ``datasets`` is installed and the network is available, and falls back
  to a small built-in synthetic corpus so every lab still runs offline.
* ``ByteTokenizer``      a zero-dependency byte-level tokenizer (vocab 256 + 3
  specials), handy for architecture ablations where tokenizer effects must be
  held constant.
* ``PackedDataset``      concatenates documents with an EOS separator and cuts
  fixed-length windows: *sequence packing*, the standard pretraining format
  that removes padding waste (Chapter 7).
"""
from __future__ import annotations

import random
from typing import Iterable, Iterator, List, Optional

import torch

_SUBJECTS = ["the small robot", "a curious cat", "the old sailor", "a young engineer",
             "the tiny model", "a quiet student", "the brave fox", "a helpful teacher"]
_VERBS = ["found", "built", "wanted", "lost", "shared", "fixed", "painted", "discovered"]
_OBJECTS = ["a red ball", "a secret map", "a broken clock", "a warm blanket",
            "a bright lamp", "a strange key", "a paper boat", "a golden leaf"]
_ENDINGS = ["and everyone was happy.", "and learned to be kind.", "so the day ended well.",
            "and they became friends.", "and it was a good lesson.", "and smiled all night."]


def synthetic_stories(n_docs: int = 20_000, seed: int = 0) -> List[str]:
    """TinyStories-flavoured synthetic text with learnable structure.

    Good enough to watch loss fall, compare architectures, and verify that a
    10M-parameter model learns grammar. Not a substitute for real data.
    """
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        s, v, o, e = (rng.choice(x) for x in (_SUBJECTS, _VERBS, _OBJECTS, _ENDINGS))
        s2, v2, o2 = rng.choice(_SUBJECTS), rng.choice(_VERBS), rng.choice(_OBJECTS)
        docs.append(f"Once upon a time, {s} {v} {o}. Then {s2} {v2} {o2}, {e}")
    return docs


def load_text_corpus(name: str = "roneneldan/TinyStories", split: str = "train",
                     max_docs: int = 50_000, text_field: str = "text") -> List[str]:
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(name, split=f"{split}[:{max_docs}]")
        return [r[text_field] for r in ds]
    except Exception as exc:  # offline, missing package, gated dataset...
        print(f"[data] falling back to synthetic corpus ({type(exc).__name__})")
        return synthetic_stories(max_docs)


class ByteTokenizer:
    """Byte-level tokenizer: ids 0-255 are raw bytes, then BOS/EOS/PAD."""
    BOS, EOS, PAD = 256, 257, 258
    vocab_size = 259

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        return ([self.BOS] if bos else []) + ids + ([self.EOS] if eos else [])

    def decode(self, ids: Iterable[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class PackedDataset(torch.utils.data.Dataset):
    """Pack documents into contiguous windows of ``seq_len + 1`` tokens.

    Inputs are window[:-1] and targets window[1:]. Documents are separated by
    EOS so the model learns boundaries; attention is NOT reset at boundaries
    (the common, slightly leaky default; see Chapter 7 for masked packing).
    """

    def __init__(self, docs: List[str], tokenizer, seq_len: int = 256,
                 max_tokens: Optional[int] = None) -> None:
        eos = getattr(tokenizer, "EOS", None)
        if eos is None:
            eos = tokenizer.eos_token_id
        stream: List[int] = []
        for d in docs:
            ids = tokenizer.encode(d)
            stream.extend(ids + [eos])
            if max_tokens and len(stream) >= max_tokens:
                break
        n = (len(stream) - 1) // seq_len
        self.data = torch.tensor(stream[: n * seq_len + 1], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, i: int):
        chunk = self.data[i * self.seq_len: (i + 1) * self.seq_len + 1]
        return chunk[:-1], chunk[1:]

    @property
    def n_tokens(self) -> int:
        return len(self.data)


def infinite_loader(ds, batch_size: int, seed: int = 0) -> Iterator:
    g = torch.Generator().manual_seed(seed)
    while True:
        for batch in torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                                 generator=g, drop_last=True):
            yield batch
