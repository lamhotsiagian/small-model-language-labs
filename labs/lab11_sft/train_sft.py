"""
Lab 11, step 3: supervised fine-tuning of a base SLM with TRL.

Small-model settings that differ from large-model defaults (Chapter 11, 11.4):
  * higher learning rate than 7B+ recipes (1e-5 to 2e-5 full FT for 0.5-1.5B)
    but very sensitive to it: sweep 3 values, do not guess
  * 2-3 epochs on a curated 50K set; watch held-out loss for overfitting
  * packing ON for throughput, with per-sample position resets
  * completion-only loss: prompt and system tokens are masked with -100
  * the SAME chat template is saved with the model and used at serving

Usage:
    python train_sft.py --base Qwen/Qwen2.5-0.5B --data data/sft_50k.jsonl --lr 2e-5 --epochs 2
    python train_sft.py --check-masking --base Qwen/Qwen2.5-0.5B    # print which tokens are trained
"""
from __future__ import annotations

import argparse

import torch


def check_masking(tok) -> None:
    """Render one example and show exactly which tokens contribute to the loss."""
    prompt = [{"role": "system", "content": "You are concise."},
              {"role": "user", "content": "Capital of Indonesia?"}]
    completion = [{"role": "assistant", "content": "Jakarta."}]
    p = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(prompt + completion, tokenize=False)
    assert full.startswith(p), "template is not prefix-consistent; masking by prefix would be wrong"
    p_ids = tok(p, add_special_tokens=False).input_ids
    f_ids = tok(full, add_special_tokens=False).input_ids
    for i, t in enumerate(f_ids):
        flag = "TRAIN" if i >= len(p_ids) else "  -  "
        print(f"{flag} {t:>7} {tok.decode([t])!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data", default="data/sft_50k.jsonl")
    ap.add_argument("--general-mix", default=None, help="optional jsonl of general data to mix in")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--check-masking", action="store_true")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    if a.check_masking:
        check_masking(tok)
        return

    from datasets import concatenate_datasets, load_dataset
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("json", data_files=a.data, split="train")
    if a.general_mix:
        ds = concatenate_datasets([ds, load_dataset("json", data_files=a.general_mix, split="train")])
    ds = ds.train_test_split(test_size=0.02, seed=0)
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    cfg = SFTConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        max_length=a.max_len,
        packing=True,                    # concatenate samples; positions reset per sample
        completion_only_loss=True,       # loss on the assistant completion only
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        report_to="none",
        seed=0,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["test"],
                         processing_class=tok)
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)           # ships the chat template WITH the weights
    print(f"[lab11] saved {a.out}")


if __name__ == "__main__":
    main()
