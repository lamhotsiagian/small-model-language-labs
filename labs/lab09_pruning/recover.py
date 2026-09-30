"""
Lab 9, step 3: recover a pruned model by distilling from its unpruned parent.

The Minitron recipe (Muralidharan et al., 2024; Sreenivas et al., 2024):
the pruned student is trained to match the ORIGINAL model's logits (forward
KL) on a modest amount of pretraining-style text. Because the student starts
from the teacher's own weights, recovery needs orders of magnitude fewer
tokens than training a model of the same size from scratch.

The script logs a recovery curve: held-out loss and KL to the teacher vs
tokens, which is the chapter's deliverable.

Usage:
    python recover.py --teacher meta-llama/Llama-3.2-3B --student checkpoints/pruned_1p5b \
        --tokens 1e9 --seq 2048 --batch 8 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab08_distillation"))
from losses import forward_kl  # noqa: E402  (shared with Lab 8)


def stream_batches(tok, seq: int, batch: int):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    buf = []
    for row in ds:
        buf.extend(tok(row["text"], add_special_tokens=False).input_ids + [tok.eos_token_id])
        while len(buf) >= seq * batch:
            chunk, buf = buf[:seq * batch], buf[seq * batch:]
            yield torch.tensor(chunk).view(batch, seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--tokens", type=float, default=1e9)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha-ce", type=float, default=0.0, help="weight of hard-label CE")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--out", default="checkpoints/recovered_1p5b")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.bfloat16).to(dev).eval()
    student = AutoModelForCausalLM.from_pretrained(a.student, torch_dtype=torch.bfloat16).to(dev)
    student.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    steps = int(a.tokens // (a.seq * a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.02,
                                                anneal_strategy="cos")
    held = [b for _, b in zip(range(8), stream_batches(tok, a.seq, a.batch))]   # first batches = held-out
    curve = []

    @torch.no_grad()
    def evaluate():
        student.eval()
        ce = kl = 0.0
        for ids in held:
            ids = ids.to(dev)
            s = student(ids).logits[:, :-1]; t = teacher(ids).logits[:, :-1]
            m = torch.ones(ids.shape[0], ids.shape[1] - 1, device=dev)
            ce += torch.nn.functional.cross_entropy(s.float().transpose(1, 2), ids[:, 1:]).item()
            kl += forward_kl(s, t, m).item()
        student.train()
        return ce / len(held), kl / len(held)

    ce0, kl0 = evaluate()
    curve.append({"tokens": 0, "heldout_ce": ce0, "kl_to_teacher": kl0})
    print(json.dumps(curve[-1]))
    for step, ids in enumerate(stream_batches(tok, a.seq, a.batch)):
        if step < len(held):
            continue                                      # never train on the held-out batches
        if step - len(held) >= steps:
            break
        ids = ids.to(dev)
        with torch.no_grad():
            t_logits = teacher(ids).logits[:, :-1]
        s_logits = student(ids).logits[:, :-1]
        mask = torch.ones(ids.shape[0], ids.shape[1] - 1, device=dev)
        loss = forward_kl(s_logits, t_logits, mask)
        if a.alpha_ce:
            loss = loss + a.alpha_ce * torch.nn.functional.cross_entropy(
                s_logits.float().transpose(1, 2), ids[:, 1:])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step(); sched.step()
        if (step - len(held) + 1) % a.eval_every == 0:
            ce, kl = evaluate()
            curve.append({"tokens": (step - len(held) + 1) * a.seq * a.batch,
                          "heldout_ce": ce, "kl_to_teacher": kl})
            print(json.dumps(curve[-1]))
    Path("results").mkdir(exist_ok=True)
    Path("results/recovery_curve.json").write_text(json.dumps(curve, indent=2))
    student.save_pretrained(a.out); tok.save_pretrained(a.out)


if __name__ == "__main__":
    main()
