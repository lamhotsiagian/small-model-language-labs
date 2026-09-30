# Lab 6: Build a Quality-Filtered Corpus

**Chapter:** 6, Pretraining Data Curation and Synthetic Data
**Goal:** Take ~5 GB of raw web text, apply heuristic filtering, PII scrubbing,
exact and MinHash dedup, train a small quality classifier from LLM
annotations, generate ~50M tokens of synthetic textbook data, decontaminate
against benchmarks, then train two 60M models (raw vs curated) and compare.

## Files

| File | Stage |
|---|---|
| `curate.py` | Gopher/C4-style heuristics, Luhn-validated PII redaction, SHA-1 exact dedup, MinHash + LSH near-dedup |
| `quality.py` | `annotate` (LLM judge, FineWeb-Edu rubric) -> `train` (TF-IDF + ridge) -> `filter` (threshold 3) |
| `synth.py` | Cosmopedia-style topic x format x audience prompt grid, vLLM or HF generation, seed tags |
| `decontam.py` | 13-gram overlap against benchmark items; flagged documents are removed |
| `train_compare.py` | Two identical 60M models, same token budget, neutral held-out loss |

## Flow

1. **Get raw text.** Stream ~5 GB of Common Crawl-derived text (for example a
   slice of `HuggingFaceFW/fineweb` *before* its own filtering, or raw WARC
   extracts) into `data/raw.jsonl` with one `{"id", "text"}` per line.
2. **Filter and dedup.** `python curate.py --input data/raw.jsonl --output data/clean.jsonl`.
   Record the per-rule drop counts: they are your first data-quality dashboard.
   (`python curate.py --demo` shows every stage on six toy documents.)
3. **Annotate a sample.** `python quality.py annotate --input data/clean.jsonl --n 50000`.
4. **Distil the judge.** `python quality.py train` prints held-out F1 at the
   threshold; aim for > 0.8 before trusting it.
5. **Filter the corpus.** `python quality.py filter --input data/clean.jsonl --output data/edu.jsonl`.
6. **Generate synthetic textbooks.** `python synth.py --n 60000 --out data/synthetic.jsonl`
   (~50M tokens at ~800 tokens each). Run the synthetic set through
   `curate.py` too: near-duplicate synthetic samples are common.
7. **Decontaminate.** `python decontam.py --train data/mix.jsonl --bench data/benchmarks.jsonl --out data/final.jsonl`.
8. **Compare.** `python train_compare.py --raw data/raw.jsonl --curated data/final.jsonl --heldout data/heldout.jsonl --tokenizer <lab03 32K>`.
   Then run lm-evaluation-harness on both checkpoints.

## Deliverable

The pipeline code plus a quality impact report: drop counts per stage, tokens
kept, classifier F1, synthetic diversity stats, contamination hits, and the
raw-vs-curated comparison (held-out loss, HellaSwag, ARC-Easy).
