# Lab 17: Serving Benchmark

**Chapter:** 17, Inference Optimization and Serving
**Goal:** Serve a 1-1.5B model on vLLM, SGLang, and llama.cpp. Load test at 1,
8, 32, 128 concurrent users, measuring TTFT, TPOT, and throughput. Then use a
0.5B model as a speculative drafter for a 7-8B target. Deliver a benchmark
report with speedup numbers.

## Files

| File | Purpose |
|---|---|
| `capacity.py` | KV-limited max concurrency per model on a 24 GB GPU; speculative-decoding expected speedup table |
| `serve.sh` | Comparable launch commands: `vllm`, `sglang`, `llamacpp`, `spec` |
| `loadtest.py` | Async streaming load generator: TTFT / TPOT / E2E percentiles, tokens/s, requests/s |

## Flow

1. **Predict first.** `python capacity.py`. Write down the concurrency at which you
   expect each engine to saturate before running anything.
2. **Serve.** `MODEL=Qwen/Qwen2.5-1.5B-Instruct bash serve.sh vllm` (then `sglang`, then
   `llamacpp` with a Q4_K_M or Q8_0 GGUF from Lab 16).
3. **Load test.** For each engine:
   `python loadtest.py --model <name> --concurrency 1 8 32 128 --duration 60 --out results/<engine>.csv`.
4. **Compare.** Plot tokens/s vs concurrency and TPOT p95 vs concurrency per engine.
   Identify the knee: where throughput stops growing and latency starts climbing.
5. **Speculative decoding.** `bash serve.sh spec` (7B target, 0.5B draft) and
   load test at concurrency 1 and 8; compare with the 7B served alone. Record the
   acceptance rate from vLLM's metrics endpoint (`/metrics`).
6. **Prefix caching.** Re-run with a long shared system prompt (edit `make_prompt`)
   with and without `--enable-prefix-caching`; compare TTFT.

## Deliverable

Benchmark report: per-engine table (tokens/s, TTFT p50/p95, TPOT p50/p95 at each
concurrency), saturation point, speculative speedup at batch 1 and 8 with
measured acceptance rate, and the prefix-caching TTFT delta.
