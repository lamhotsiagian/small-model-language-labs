"""
Lab 14, step 3: one task, five adaptation methods, one measurement harness.

  --method full    full fine-tuning (all weights, bf16 + fp32 AdamW states)
  --method lora    LoRA on all linear layers (--r 8 or 64), bf16 base
  --method qlora   LoRA on a 4-bit NF4 base with double quantisation and a
                   paged 8-bit optimizer (Dettmers et al., 2023)
  --method dora    weight-decomposed LoRA: magnitude vector + LoRA direction
                   (S.-Y. Liu et al., 2024)

Every run logs peak GPU memory, wall-clock, and test accuracy to
results/<run>.json so plot_frontier.py can draw the cost-quality frontier.

Usage:
    python train_peft.py --method lora --r 8
    python train_peft.py --method lora --r 64
    python train_peft.py --method qlora --r 64
    python train_peft.py --method dora --r 8
    python train_peft.py --method full --lr 1e-5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from make_security_data import LABELS


@torch.no_grad()
def accuracy(model, tok, rows, batch: int = 32) -> float:
    model.eval()
    tok.padding_side = "left"
    correct = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        prompts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        for o, r in zip(out, chunk):
            pred = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip().split()
            correct += bool(pred) and pred[0].strip(".,") == r["label"]
    return correct / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["full", "lora", "qlora", "dora"], required=True)
    ap.add_argument("--base", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=None, help="default 2r")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    run = f"{a.method}" + ("" if a.method == "full" else f"_r{a.r}")
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    kw = dict(torch_dtype=torch.bfloat16)
    if a.method == "qlora":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.base, device_map={"": 0}, **kw)
    if a.method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_cfg = None
    if a.method != "full":
        peft_cfg = LoraConfig(r=a.r, lora_alpha=a.alpha or 2 * a.r, lora_dropout=0.05, bias="none",
                              target_modules="all-linear", use_dora=(a.method == "dora"),
                              task_type="CAUSAL_LM")
    lr = a.lr or (1e-5 if a.method == "full" else 2e-4)
    ds = load_dataset("json", data_files={s: f"{a.data}/{s}.jsonl" for s in ("train", "val")})
    cfg = SFTConfig(output_dir=f"checkpoints/{run}", num_train_epochs=a.epochs, learning_rate=lr,
                    per_device_train_batch_size=16, gradient_accumulation_steps=2, lr_scheduler_type="cosine",
                    warmup_ratio=0.03, bf16=True, gradient_checkpointing=True, max_length=512,
                    completion_only_loss=True, logging_steps=20, save_strategy="no", report_to="none",
                    optim="paged_adamw_8bit" if a.method == "qlora" else "adamw_torch_fused")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["val"],
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    train_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    test = [json.loads(l) for l in open(f"{a.data}/test.jsonl")]
    acc = accuracy(trainer.model, tok, test)
    trainer.save_model(f"checkpoints/{run}")
    res = {"run": run, "method": a.method, "r": a.r, "lr": lr, "trainable_params": trainable,
           "peak_gb": round(peak, 2), "train_minutes": round(train_s / 60, 1), "test_acc": round(acc, 4)}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{run}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
