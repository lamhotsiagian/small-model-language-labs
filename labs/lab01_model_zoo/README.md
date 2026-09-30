# Lab 1: SLM Model Zoo Benchmark

**Chapter:** 1, The SLM Landscape and When Small Wins
**Goal:** Run five small models through the same 50 prompts across five task
types, record latency, memory, and quality, and produce a decision matrix that
recommends one model per task.

## Files

| File | Purpose |
|---|---|
| `tasks.py` | 50 prompts (classification, extraction, routing, summarization, tool calling) with automatic scorers in [0, 1] |
| `benchmark.py` | Loads each model, applies its chat template, times TTFT and TPOT, scores outputs |
| `decision_matrix.py` | Applies a quality floor and memory ceiling, ranks models per task, prints a TCO comparison |

## Flow

1. **Pin the zoo.** Edit `DEFAULT_ZOO` in `benchmark.py`. Accept licenses for
   gated models (Llama, Gemma) on the Hub and run `huggingface-cli login`.
   Record the commit SHA of each model for reproducibility.
2. **Smoke test.** `python benchmark.py --preset smoke` runs two tiny models on
   10 prompts so you can check the pipeline in minutes on a CPU.
3. **Full run.** `python benchmark.py` runs every model on all 50 prompts with
   greedy decoding (temperature 0) and `max_new_tokens=96`.
   Output: `results/zoo_results.json`.
4. **Decide.** `python decision_matrix.py --quality-floor 0.8 --memory-ceiling-mb 2500`
   ranks models per task. Output: `results/decision_matrix.md`.
5. **Interpret.** For each task type, write one sentence explaining *why* the
   winner won (for example "routing is a 2-label decision; 360M is enough").
   Flag every task where no model clears the floor: those are candidates for
   fine-tuning (Chapters 11 and 14) or escalation to a larger model.

## Deliverable

A decision matrix recommending a model per task type, with quality, p50 TTFT,
p50 TPOT, weights footprint, and a one-line justification per row.

## What to observe

- Classification and routing saturate early: sub-1B models often match 3B.
- Extraction and tool calling depend on format discipline more than knowledge;
  scores jump when you add constrained decoding (Lab 15).
- Latency differences between models are dominated by decode (TPOT x tokens),
  not by prefill, for short prompts.
