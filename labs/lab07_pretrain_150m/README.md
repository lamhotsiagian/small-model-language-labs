# Lab 7: Pretrain a 150M SLM

**Chapter:** 7, Pretraining an SLM From Scratch
**Goal:** Pretrain a Llama-style ~151M model on 3-5B tokens on a single
multi-GPU node (or rented GPUs). Track MFU, loss, and periodic HellaSwag.
Deliver a trained checkpoint, a training report, and a dashboard.

## Files

| File | Purpose |
|---|---|
| `config_150m.yaml` | Model (d=768, 18 layers, GQA 12/4, SwiGLU 2048, QK-norm, 49K tied vocab) and training (WSD, 1M-token batch, z-loss) |
| `prepare_data.py` | Streams FineWeb-Edu, tokenizes once into uint16 memmap shards plus a validation shard |
| `pretrain.py` | DDP or FSDP via `torchrun`; BF16, compile, grad accumulation, spike guard, MFU, atomic resumable checkpoints, HellaSwag scorer |
| `../../slmlab/export.py` | Converts the checkpoint to HF Llama/Qwen3 format (logits match to ~1e-7) for lm-eval, vLLM, llama.cpp |

## Flow

1. **Tokenize.** `python prepare_data.py --tokens 5e9 --out data/` (~30-60 min on a
   CPU box; do it once, reuse for every run).
2. **Smoke test.** `python pretrain.py --smoke` runs the entire code path on a
   CPU with a 0.9M model and synthetic shards (about a minute).
3. **Launch.** `torchrun --nproc_per_node=8 pretrain.py --config config_150m.yaml`.
   With a ~1M-token global batch, 5B tokens is ~4,770 steps. On 8 x H100 at
   ~30% MFU this is under an hour; on 8 x A100 about two hours; on 1 x 4090 about a day.
4. **Watch.** Every 500 steps the job logs loss, validation loss, grad norm,
   tokens/s, and MFU as JSON lines. Pipe them to W&B or TensorBoard, or plot
   `checkpoints/slm150m/train_log.json`.
5. **Kill and resume.** Stop the job mid-run and relaunch the same command:
   it resumes from the latest atomic checkpoint. Verify the loss curve is continuous.
6. **Evaluate.** Export and run the harness:
   ```bash
   python -m slmlab.export --ckpt checkpoints/slm150m/step_XXXXXXX.pt \
       --config labs/lab07_pretrain_150m/config_150m.yaml \
       --tokenizer HuggingFaceTB/SmolLM2-135M --out exported/slm150m
   lm_eval --model hf --model_args pretrained=exported/slm150m --tasks hellaswag,arc_easy,piqa
   ```
7. **Report.** Final val loss, HellaSwag acc_norm, tokens/s, MFU, wall-clock,
   GPU-hours, and any spikes or restarts, with the loss curve.

## Reference output (smoke preset, CPU)

```
[lab07] params=0.9M world=1 accum=2 steps=97 tokens/step=2,048
{"step": 0,  "loss": 6.2811, "val_loss": 6.2419, "grad_norm": 0.799, ...}
{"step": 96, "loss": 1.3338, "val_loss": 1.305,  "grad_norm": 0.388, ...}
```
