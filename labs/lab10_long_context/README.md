# Lab 10: Extend Context 4x

**Chapter:** 10, Long Context for Small Models
**Goal:** Extend a 1-2B model from 8K to 32K using YaRN plus short continued
pretraining. Evaluate with RULER-style probes at 4K, 8K, 16K, 32K. Deliver an
accuracy-by-length chart and a memory profile.

## Files

| File | Purpose |
|---|---|
| `rope_scaling_demo.py` | Wavelength table for original / PI / NTK / YaRN (uses `slmlab.model.rope_frequencies`) |
| `extend.py` | Sets YaRN `rope_scaling` in the HF config, then continued pretraining on 32K packed sequences with per-document `position_ids` (document masking with FlashAttention-2) |
| `ruler_lite.py` | NIAH single / multi-key / multi-value and variable tracking at any length and needle depth; `--demo` prints examples |
| `kv_profile.py` | Weights + KV cache vs context for fp16/int8/int4 KV, a StreamingLLM window, and a GQA counterfactual |

Default model: `HuggingFaceTB/SmolLM2-1.7B` (8,192 native context, full MHA).

## Flow

1. **See the mechanism.** `python rope_scaling_demo.py --factor 4`.
2. **Zero-shot YaRN.** `python extend.py --no-train --out checkpoints/yarn_zeroshot` just
   edits the config. Run `ruler_lite.py` on it: expect partial success beyond 8K.
3. **Continued pretraining.** `python extend.py --tokens 5e8 --long-frac 0.7` (one 80 GB
   GPU, gradient checkpointing; roughly 15K steps of 32K tokens).
4. **Probe.** `python ruler_lite.py --model checkpoints/smollm2_1p7b_32k --lengths 4096 8192 16384 32768`
   for base, zero-shot YaRN, and YaRN+CPT. Plot accuracy vs length per task.
5. **Profile memory.** `python kv_profile.py --arch SmolLM2-1.7B` and compare with
   `torch.cuda.max_memory_allocated()` during a 32K prefill.
6. **Guard short context.** Re-run HellaSwag / ARC on the extended model: long-context
   training must not cost short-context quality (the `--long-frac` knob).

## Deliverable

Accuracy-by-length chart (3 models x 4 tasks x 4 lengths), memory profile
table, and short-context regression check.

## Reference output (`python kv_profile.py`, computed)

```
SmolLM2-1.7B: weights 3.19 GB (bf16), KV heads 32, KV/token 192 KB (fp16)
 context   fp16 KV   int8 KV   int4 KV  stream 2K  GQA/4 fp16
   32768     6.00G     3.00G     1.50G      0.38G       1.50G
```
