"""
Lab 8, step 2: three ways to distil a 7-8B teacher into a 0.5B student.

  --method sft     sequence-level KD: cross-entropy on correct teacher traces
  --method logit   token-level KD: forward KL to the teacher's distribution on
                   the same traces (online teacher, or offline top-k with --offline)
  --method gkd     on-policy KD: the STUDENT samples responses, the teacher scores
                   every token, loss = generalised JSD (Agarwal et al., 2024).
                   `--lam` is the fraction of batches that are on-policy.

Teacher and student must share a tokenizer for logit and GKD methods (e.g.
Qwen2.5-7B-Instruct -> Qwen2.5-0.5B-Instruct, or Llama-3.1-8B-Instruct ->
Llama-3.2-1B-Instruct). See Chapter 8, section 8.5 for mismatched tokenizers.

Usage:
    python train_kd.py --method sft   --data data/teacher.jsonl
    python train_kd.py --method logit --data data/teacher.jsonl --tau 1.0
    python train_kd.py --method gkd   --data data/teacher.jsonl --beta 0.5 --lam 0.5
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from gsm8k import messages
from losses import forward_kl, gkd_jsd, sft_ce, topk_kl


def encode(tok, question: str, response: str, max_len: int = 1024):
    prompt = tok.apply_chat_template(messages(question), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    r_ids = tok(response + tok.eos_token, add_special_tokens=False).input_ids
    ids = (p_ids + r_ids)[:max_len]
    mask = ([0] * len(p_ids) + [1] * len(r_ids))[:max_len]   # distil response tokens only
    return ids, mask


def collate(batch, pad_id: int, device):
    L = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), L), pad_id)
    msk = torch.zeros(len(batch), L)
    for j, (i, m) in enumerate(batch):
        ids[j, :len(i)] = torch.tensor(i)
        msk[j, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids.to(device), msk.to(device)


@torch.no_grad()
def student_rollouts(student, tok, questions, max_new: int = 384, temperature: float = 1.0):
    """On-policy data: the student's own samples for the batch's prompts."""
    prompts = [tok.apply_chat_template(messages(q), tokenize=False, add_generation_prompt=True)
               for q in questions]
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(student.device)
    out = student.generate(**enc, do_sample=True, temperature=temperature, max_new_tokens=max_new,
                           pad_token_id=tok.pad_token_id)
    tok.padding_side = "right"
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sft", "logit", "gkd"], required=True)
    ap.add_argument("--data", default="data/teacher.jsonl")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--offline", action="store_true", help="use stored top-k instead of a live teacher")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.student)
    tok.pad_token = tok.pad_token or tok.eos_token
    student = AutoModelForCausalLM.from_pretrained(a.student, torch_dtype=torch.bfloat16).to(dev)
    teacher = None
    if a.method in ("logit", "gkd") and not a.offline:
        teacher = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.bfloat16).to(dev).eval()
    rows = [json.loads(l) for l in open(a.data)]
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0)
    step, t0, teacher_fwd = 0, time.time(), 0

    for epoch in range(a.epochs):
        random.Random(epoch).shuffle(rows)
        for i in range(0, len(rows) - a.batch + 1, a.batch):
            chunk = rows[i:i + a.batch]
            on_policy = a.method == "gkd" and random.random() < a.lam
            responses = (student_rollouts(student, tok, [r["question"] for r in chunk])
                         if on_policy else [r["response"] for r in chunk])
            ids, mask = collate([encode(tok, r["question"], resp) for r, resp in zip(chunk, responses)],
                                tok.pad_token_id, dev)
            s_logits = student(ids).logits[:, :-1]
            m, labels = mask[:, 1:], ids[:, 1:]

            if a.method == "sft":
                loss = sft_ce(s_logits, labels, m)
            elif a.offline:
                # stored top-k teacher: ids/logps aligned to response positions
                k_ids, k_lp = offline_topk(chunk, ids.shape[1] - 1, m, dev)
                loss = topk_kl(s_logits, k_ids, k_lp, m, a.tau)
            else:
                with torch.no_grad():
                    t_logits = teacher(ids).logits[:, :-1]
                # Same tokenizer, different padded vocab: Qwen2.5-7B has 152,064 logit
                # rows, 0.5B has 151,936. The extra rows are unused padding; drop them.
                t_logits = t_logits[..., :s_logits.shape[-1]]
                teacher_fwd += 1
                loss = (forward_kl(s_logits, t_logits, m, a.tau) if a.method == "logit"
                        else gkd_jsd(s_logits, t_logits, m, a.beta, a.tau))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 25 == 0:
                print(json.dumps({"step": step, "loss": round(loss.item(), 4), "on_policy": on_policy,
                                  "teacher_fwd": teacher_fwd, "elapsed_s": int(time.time() - t0)}))
    out = a.out or f"checkpoints/student_{a.method}"
    student.save_pretrained(out); tok.save_pretrained(out)
    print(f"[lab08] saved {out}; GPU-seconds {int(time.time() - t0)}; teacher forwards {teacher_fwd}")


def offline_topk(chunk, T, mask, dev):
    """Scatter stored per-token top-k into [B, T, k] tensors aligned with the mask."""
    k = len(chunk[0]["topk"][0])
    ids = torch.zeros(len(chunk), T, k, dtype=torch.long, device=dev)
    lps = torch.full((len(chunk), T, k), -1e4, device=dev)
    for b, r in enumerate(chunk):
        pos = mask[b].nonzero().flatten().tolist()
        for p, entry in zip(pos, r["topk"]):
            ids[b, p] = torch.tensor([t for t, _ in entry], device=dev)
            lps[b, p] = torch.tensor([lp for _, lp in entry], device=dev)
    return ids, lps


if __name__ == "__main__":
    main()
