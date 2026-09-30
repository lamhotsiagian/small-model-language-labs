"""
Lab 5, steps 1-2: train a grid of small models for a scaling-law fit.

Grid: 4 model sizes (5M, 15M, 40M, 100M non-embedding + embedding) x 3 token
budgets, plus a held-out 250M run used only to test the prediction.

Compute trick (Chapter 5, section 5.4): with a warmup-stable-decay (WSD)
schedule you do NOT need one run per token budget. Train each size once at
constant LR, snapshot the model at each budget's decay start, and branch a
short linear decay from every snapshot. Three budgets cost ~1.2x one run
instead of ~2x (Hu et al., 2024; Hagele et al., 2024).

Usage:
    python sweep.py --preset smoke                   # 3 tiny sizes x 2 budgets, CPU
    python sweep.py --preset full --device cuda      # the chapter grid
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.budget import ArchSpec, param_breakdown
from slmlab.data import ByteTokenizer, PackedDataset, infinite_loader, load_text_corpus
from slmlab.model import SLMConfig, SmallLM
from slmlab.train import TrainConfig, build_optimizer, evaluate


def config_for(target: int, vocab: int, aspect: int = 48) -> SLMConfig:
    """Pick (d, L) with d/L ~ aspect and heads of 64 dims so params ~ target."""
    best = None
    for d in range(64, 2049, 64):
        L = max(2, round(d / aspect))
        h = max(2, d // 64)
        kv = max(1, h // 4) if h % 4 == 0 else h
        dff = int(round(8 / 3 * d / 64) * 64)
        n = param_breakdown(ArchSpec("x", d, L, h, kv, dff, vocab))["total"]
        if best is None or abs(n - target) < abs(best[0] - target):
            best = (n, SLMConfig(vocab_size=vocab, d_model=d, n_layers=L, n_heads=h,
                                 n_kv_heads=kv, d_ff=dff))
    return best[1]


def wsd_branches(model, loader, val_loader, budgets_tokens, tokens_per_step, lr, device,
                 warmup=100, decay_frac=0.2, eval_batches=20):
    """One stable run, one decay branch per budget. Returns [(tokens, val_loss)]."""
    opt = build_optimizer(model, TrainConfig(lr=lr))
    total_steps = [int(b / tokens_per_step) for b in budgets_tokens]
    branch_points = sorted(int(s * (1 - decay_frac)) for s in total_steps)
    results, step = [], 0
    model.to(device).train()
    for bp, final in zip(branch_points, sorted(total_steps)):
        while step < bp:                                   # stable phase (shared)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / warmup)
            x, y = next(loader)
            _, loss = model(x.to(device), y.to(device))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
        # branch: copy model + optimizer, decay linearly to 10% LR
        m2, o2 = copy.deepcopy(model), None
        o2 = build_optimizer(m2, TrainConfig(lr=lr)); o2.load_state_dict(opt.state_dict())
        for s in range(bp, final):
            frac = (s - bp) / max(1, final - bp)
            for g in o2.param_groups:
                g["lr"] = lr * (1 - 0.9 * frac)
            x, y = next(loader)
            _, loss = m2(x.to(device), y.to(device))
            o2.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0); o2.step()
        results.append((final * tokens_per_step, evaluate(m2, val_loader, eval_batches, device)))
        del m2, o2
    return results


PRESETS = {
    "smoke": dict(sizes=[2e5, 6e5, 1.5e6], budgets=[2e5, 6e5], seq=128, batch=16, docs=8000, lr=3e-3),
    "full": dict(sizes=[5e6, 15e6, 40e6, 100e6, 250e6], budgets=[0.2e9, 0.6e9, 2e9],
                 seq=1024, batch=64, docs=3_000_000, lr=2e-3),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="smoke")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/sweep.json")
    a = ap.parse_args()
    P = PRESETS[a.preset]
    tok = ByteTokenizer()
    docs = load_text_corpus(max_docs=P["docs"])
    split = int(0.98 * len(docs))
    tr = PackedDataset(docs[:split], tok, P["seq"])
    va = PackedDataset(docs[split:], tok, P["seq"])
    tps = P["seq"] * P["batch"]

    runs = []
    for target in P["sizes"]:
        cfg = config_for(int(target), tok.vocab_size)
        cfg.max_seq_len = P["seq"]
        torch.manual_seed(0)
        model = SmallLM(cfg)
        # muP-lite: scale LR by base_width / width so wider models do not diverge.
        lr = P["lr"] * min(1.0, 256 / cfg.d_model)
        print(f"[lab05] N={model.num_params():,} d={cfg.d_model} L={cfg.n_layers} lr={lr:.2e}")
        for tokens, loss in wsd_branches(model, infinite_loader(tr, P["batch"]),
                                         infinite_loader(va, P["batch"], seed=1),
                                         P["budgets"], tps, lr, a.device):
            runs.append({"params": model.num_params(), "tokens": tokens, "val_loss": loss})
            print(f"   D={tokens:,.0f}  val_loss={loss:.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
