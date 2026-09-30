# Lab 14: PEFT Showdown

**Chapter:** 14, Parameter-Efficient Fine-Tuning
**Goal:** Fine-tune a 3B model on security-log classification with full FT,
LoRA r=8 and r=64, QLoRA, and DoRA. Compare accuracy, VRAM, and time. Serve
three adapters from one base in vLLM. Deliver a cost-quality frontier chart.

## Files

| File | Purpose |
|---|---|
| `lora_math.py` | LoRA layer from scratch (B=0 init, merge equivalence) + memory calculator for Llama-3.2-3B |
| `make_security_data.py` | 8-class synthetic SOC log dataset with class imbalance, hard negatives, label noise |
| `train_peft.py` | `--method full | lora | qlora | dora`, logs peak VRAM, minutes, test accuracy |
| `plot_frontier.py` | VRAM vs accuracy frontier + Markdown table |
| `serve_multilora.sh`, `client.py` | vLLM multi-LoRA server and a concurrent mixed-adapter client |
| `merge.py` | Fold an adapter into the base, or combine adapters with linear / TIES / DARE |

## Flow

1. **Math first.** `python lora_math.py` (CPU): verify step-0 equivalence and read the memory table.
2. **Data.** `python make_security_data.py --n 12000 --noise 0.02`.
3. **Train five variants** (one 24 GB GPU for LoRA/QLoRA/DoRA; 80 GB for full FT):
   ```bash
   python train_peft.py --method lora --r 8
   python train_peft.py --method lora --r 64
   python train_peft.py --method qlora --r 64
   python train_peft.py --method dora --r 8
   python train_peft.py --method full
   ```
4. **Frontier.** `python plot_frontier.py` -> `results/frontier.png` and a table.
5. **Serve.** `bash serve_multilora.sh` then `python client.py --n 200`: three adapters,
   one base, concurrent traffic. Compare memory with three merged models.
6. **Merge (optional).** `python merge.py --adapter checkpoints/lora_r8 --merge-only`.

## Deliverable

Frontier chart and table (trainable params, peak GB, minutes, test accuracy)
plus multi-LoRA serving latency per adapter.

## Reference output (`python lora_math.py`, computed)

```
method                   trainable   share    memory
full fine-tune              3213M    100%    47.9GB
LoRA r=8 (all linear)       12.2M   0.38%     6.2GB
LoRA r=64 (all linear)      97.3M   3.03%     7.4GB
QLoRA r=8                   12.2M   0.38%     2.3GB
QLoRA r=64                  97.3M   3.03%     3.5GB
```
