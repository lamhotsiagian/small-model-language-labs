"""
Lab 12, step 2: align the Lab 11 SFT model with DPO, ORPO, or SimPO (TRL).

  --method dpo    DPOTrainer, loss_type="sigmoid", frozen reference = SFT model
  --method ipo    DPOTrainer, loss_type="ipo"
  --method simpo  CPOTrainer, loss_type="simpo", cpo_alpha=0 (reference-free)
  --method orpo   ORPOTrainer (reference-free, adds NLL on chosen; can start from BASE)

Small-model notes (Chapter 12, 12.6):
  * lower LR than SFT (5e-7 to 1e-6 for DPO full FT)
  * beta 0.05-0.1 for DPO; SimPO beta ~2, gamma/beta ~0.5
  * one epoch; watch chosen/rejected reward curves and response LENGTH

Usage:
    python train_pref.py --method dpo   --policy ../lab11_sft/checkpoints/sft --data data/pairs.jsonl
    python train_pref.py --method simpo --policy ../lab11_sft/checkpoints/sft --data data/pairs.jsonl
    python train_pref.py --method orpo  --policy ../lab11_sft/checkpoints/sft --data data/pairs.jsonl
"""
from __future__ import annotations

import argparse

import torch

DEFAULTS = {  # method: (lr, extra config)
    "dpo": (1e-6, dict(beta=0.1, loss_type="sigmoid")),
    "ipo": (1e-6, dict(beta=0.1, loss_type="ipo")),
    "simpo": (1e-6, dict(beta=2.0, simpo_gamma=1.0, loss_type="simpo", cpo_alpha=0.0)),
    "orpo": (8e-6, dict(beta=0.1)),            # in TRL's ORPO, beta is the lambda on the odds ratio
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=DEFAULTS, required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--data", default="data/pairs.jsonl")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import CPOConfig, CPOTrainer, DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer

    lr, extra = DEFAULTS[a.method]
    lr = a.lr or lr
    out = a.out or f"checkpoints/{a.method}"
    tok = AutoTokenizer.from_pretrained(a.policy)
    model = AutoModelForCausalLM.from_pretrained(a.policy, torch_dtype=torch.bfloat16)
    ds = load_dataset("json", data_files=a.data, split="train").train_test_split(test_size=0.03, seed=0)
    common = dict(output_dir=out, learning_rate=lr, num_train_epochs=a.epochs,
                  per_device_train_batch_size=4, gradient_accumulation_steps=8,
                  lr_scheduler_type="cosine", warmup_ratio=0.1, bf16=True,
                  gradient_checkpointing=True, max_length=2048, logging_steps=10,
                  eval_strategy="steps", eval_steps=100, report_to="none", seed=0)
    if a.method in ("dpo", "ipo"):
        # ref_model=None -> TRL makes a frozen copy of the starting policy as the reference.
        trainer = DPOTrainer(model=model, ref_model=None, args=DPOConfig(**common, **extra),
                             train_dataset=ds["train"], eval_dataset=ds["test"], processing_class=tok)
    elif a.method == "simpo":
        trainer = CPOTrainer(model=model, args=CPOConfig(**common, **extra),
                             train_dataset=ds["train"], eval_dataset=ds["test"], processing_class=tok)
    else:
        trainer = ORPOTrainer(model=model, args=ORPOConfig(**common, **extra),
                              train_dataset=ds["train"], eval_dataset=ds["test"], processing_class=tok)
    trainer.train()
    trainer.save_model(out); tok.save_pretrained(out)
    print(f"[lab12] saved {out}")


if __name__ == "__main__":
    main()
