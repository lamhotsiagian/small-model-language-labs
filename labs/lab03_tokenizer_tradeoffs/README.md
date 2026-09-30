# Lab 3: Tokenizer Trade-off Study

**Chapter:** 3, Tokenization and Vocabulary Design
**Goal:** Train byte-level BPE tokenizers at 16K, 32K, and 64K vocabulary on a
mixed English, Indonesian, and code corpus. Measure fertility per language
and compute the embedding parameter share for a ~300M model. Recommend a
vocabulary size with justification.

## Files

| File | Purpose |
|---|---|
| `build_corpus.py` | Streams a fixed character budget per language (FineWeb-Edu, Indonesian Wikipedia, the-stack-smol) and writes train / held-out splits |
| `train_tokenizers.py` | Byte-level BPE with digit splitting and reserved chat/tool special tokens |
| `analyze.py` | Fertility and bytes/token per language, embedding share on a fixed 24-layer d=1024 backbone, and a recommendation rule |
| `sample_corpus/` | Tiny offline corpus so the pipeline runs without network (toy results only) |

## Flow

1. **Build the corpus.** `python build_corpus.py --chars-per-lang 50_000_000 --mix en=0.5 id=0.3 code=0.2`.
   Keep the held-out split: fertility must be measured on text the tokenizer did not train on.
2. **Train tokenizers.** `python train_tokenizers.py --vocab-sizes 16000 32000 64000`.
   Optionally repeat with `--no-digit-split` to see the effect on numbers.
3. **Measure fertility.** `python analyze.py` prints tokens per word and
   bytes per token for each language and each vocabulary.
4. **Price the vocabulary.** The same script computes total parameters and
   embedding share for the fixed ~300M backbone (tied embeddings).
5. **Recommend.** The rule picks the smallest vocabulary whose worst-language
   fertility is within 5% of the best and whose embedding share is under 20%.
   Write a one-paragraph justification naming the language that drove the choice.
6. **Ablate the mixture.** Rebuild with `--mix en=0.8 id=0.1 code=0.1` and
   observe Indonesian fertility rise: the mixture, not only the size, decides fairness.

## Deliverable

`results/tokenizer_report.json` and a recommended vocabulary size with a
written justification.

## Offline smoke test

```bash
python build_corpus.py --offline
python train_tokenizers.py --vocab-sizes 500 1000 2000
python analyze.py --max-embed-share 0.5
```
The toy corpus saturates below 1K merges; use it only to check the pipeline.
