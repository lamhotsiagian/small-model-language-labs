# Lab 18: Ship an SLM to a Phone or Laptop

**Chapter:** 18, On-Device and Edge Deployment
**Goal:** Convert your fine-tuned model to GGUF (llama.cpp) and one mobile
runtime (MLC or ExecuTorch). Run it on a real device and measure tokens/sec,
memory, and battery drain over 10 minutes. Deliver a working on-device demo
plus a performance profile.

## Files

| File | Purpose |
|---|---|
| `memory_plan.py` | Weights + KV + activations + overhead vs realistic app memory budgets; bandwidth roofline per device |
| `export_mobile.sh` | `gguf` (llama.cpp), `mlc` (MLC-LLM for Android/iOS/WebGPU), `executorch` (.pte, XNNPACK, 8da4w) |
| `sustained_bench.py` | 10-minute continuous generation: per-minute tokens/s, RSS, battery %, temperature (local or `adb`) |
| `hybrid_router.py` | On-device vs cloud routing policy from privacy, connectivity, length, confidence, battery, thermals |

## Flow

1. **Plan.** `python memory_plan.py --context 4096`. Pick the largest model/format that
   fits your device's *app* budget with headroom.
2. **Export.** `MODEL=../lab11_sft/checkpoints/sft bash export_mobile.sh gguf`, then
   `mlc` (Android/iOS) or `executorch` (Llama-family checkpoints).
3. **Run on the device.** Laptop / Pi / Jetson: `llama-cli -m mobile/model-Q4_K_M.gguf`.
   Android: MLC's sample app or an ExecuTorch demo app with the exported artifact.
4. **Sustained benchmark.** `python sustained_bench.py --backend llama_cpp --gguf mobile/model-Q4_K_M.gguf --minutes 10`
   (or `--backend adb` on Android). Record minute 1 vs minute 10 tokens/s, battery used, peak temperature.
5. **Hybrid policy.** `python hybrid_router.py`; adapt thresholds to your app and log decisions.
6. **Compare runtimes.** Same model, same prompt, GGUF vs MLC/ExecuTorch on the same device.

## Deliverable

A demo (screen recording or terminal session) plus a profile: model size,
peak RSS, TTFT, tokens/s at minute 1 and minute 10, battery %/10 min, peak
temperature, per runtime.
