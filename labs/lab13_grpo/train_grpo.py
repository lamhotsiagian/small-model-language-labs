"""
Lab 13: GRPO on a 1.5B model with verifiable math rewards (TRL GRPOTrainer).

Defaults follow what works for small models (Chapter 13, 13.3):
  * start from an INSTRUCT model (or a lightly distilled one): a base 1.5B model
    rarely produces a correct, parsable answer, and GRPO learns nothing from a
    group where every reward is 0
  * G = 8 samples per prompt, temperature 1.0, max 512 new tokens
  * LR 1e-6, no KL term (beta = 0), clipped importance ratios
  * filter prompts the model already solves 8/8 or 0/8 (zero advantage)

Usage:
    python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 500 --use-vllm
    python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --loss-type dr_grpo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab08_distillation"))
from gsm8k import SYSTEM, gold  # noqa: E402
from rewards import correctness_reward, format_reward, length_penalty  # noqa: E402

THINK_SYSTEM = (SYSTEM + " First reason inside <think> ... </think>, then give the final "
                "answer on its own line as '#### <number>'.")


def load_prompts(n_math: int = 0):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.map(lambda r: {"prompt": [{"role": "system", "content": THINK_SYSTEM},
                                      {"role": "user", "content": r["question"]}],
                           "answer": gold(r["answer"])}, remove_columns=ds.column_names)
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--loss-type", default="grpo", choices=["grpo", "dr_grpo", "dapo"])
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--out", default="checkpoints/grpo")
    a = ap.parse_args()
    from trl import GRPOConfig, GRPOTrainer

    cfg = GRPOConfig(
        output_dir=a.out,
        max_steps=a.steps,
        learning_rate=a.lr,
        beta=a.beta,                       # KL to reference; 0 drops the reference model
        loss_type=a.loss_type,
        num_generations=a.group,           # G samples per prompt share one baseline
        per_device_train_batch_size=a.group * 2,
        gradient_accumulation_steps=4,
        max_completion_length=a.max_new,
        temperature=1.0,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=a.use_vllm,
        logging_steps=5,
        log_completions=True,              # sample completions in the logs for inspection
        save_steps=100,
        report_to="none",
        seed=0,
    )
    trainer = GRPOTrainer(model=a.model, args=cfg, train_dataset=load_prompts(),
                          reward_funcs=[correctness_reward, format_reward, length_penalty])
    trainer.train()
    trainer.save_model(a.out)
    print(f"[lab13] saved {a.out}; logs in {a.out}/trainer_state.json")


if __name__ == "__main__":
    main()
