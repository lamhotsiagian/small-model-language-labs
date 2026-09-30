"""
slmlab.train
============

One small, honest training loop reused by Labs 2, 4, 5, and 7. It implements
the production habits Chapter 7 argues for, at a scale that runs on a laptop:

* AdamW with decoupled weight decay applied only to matrices (not norms/embeds)
* warmup-stable-decay (WSD) or cosine schedules (Chapter 5)
* bf16 autocast on GPU, gradient accumulation, global-norm clipping
* optional z-loss and loss-spike detection with checkpoint rollback
* tokens/sec and model-FLOPs-utilisation (MFU) accounting
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import torch

from .budget import flops_per_token


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup: int = 100
    schedule: str = "wsd"          # "wsd" | "cosine" | "constant"
    decay_frac: float = 0.2        # WSD: final fraction of steps spent decaying
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    clip: float = 1.0
    z_loss: float = 0.0
    eval_every: int = 200
    eval_batches: int = 20
    spike_factor: float = 3.0      # loss > factor * EMA triggers rollback
    peak_tflops: float = 0.0       # hardware peak for MFU (e.g. 989 for H100 bf16 dense)
    seed: int = 0
    out_dir: Optional[str] = None


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate multiplier schedule (returns absolute LR)."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    if cfg.schedule == "constant":
        return cfg.lr
    min_lr = cfg.lr * cfg.min_lr_ratio
    if cfg.schedule == "cosine":
        p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return min_lr + 0.5 * (cfg.lr - min_lr) * (1 + math.cos(math.pi * p))
    # WSD: hold the peak LR, then decay linearly over the last decay_frac.
    decay_start = int(cfg.steps * (1 - cfg.decay_frac))
    if step < decay_start:
        return cfg.lr
    p = (step - decay_start) / max(1, cfg.steps - decay_start)
    return cfg.lr - (cfg.lr - min_lr) * p


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 2-D weight matrices get weight decay; norms, biases, embeddings do not.
        (decay if p.dim() >= 2 and "embed" not in n else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, fused=fused)


@torch.no_grad()
def evaluate(model, loader, n_batches: int, device) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = next(loader)
        _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(model, train_loader, val_loader, cfg: TrainConfig,
          device: str = "cpu", log: Callable[[dict], None] = print) -> list:
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    opt = build_optimizer(model, cfg)
    use_amp = device.startswith("cuda")
    n_params = model.num_params()
    history, ema, good_state = [], None, None
    t0, tokens = time.time(), 0

    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = next(train_loader)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, y, z_loss=cfg.z_loss)
            (loss / cfg.grad_accum).backward()
            step_loss += loss.item() / cfg.grad_accum
            tokens += x.numel()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip).item()

        # Loss-spike guard: skip the update and roll back to the last good state.
        if ema is not None and step_loss > cfg.spike_factor * ema and good_state is not None:
            log({"step": step, "event": "spike_rollback", "loss": step_loss, "ema": ema})
            model.load_state_dict(good_state["model"])
            opt.load_state_dict(good_state["opt"])
            continue
        opt.step()
        ema = step_loss if ema is None else 0.98 * ema + 0.02 * step_loss

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            dt = time.time() - t0
            tps = tokens / dt
            rec = {"step": step, "lr": opt.param_groups[0]["lr"], "train_loss": round(step_loss, 4),
                   "val_loss": round(evaluate(model, val_loader, cfg.eval_batches, device), 4),
                   "grad_norm": round(gnorm, 3), "tokens": tokens, "tok_per_s": round(tps)}
            if cfg.peak_tflops:
                rec["mfu"] = round(flops_per_token(n_params, training=True) * tps
                                   / (cfg.peak_tflops * 1e12), 4)
            history.append(rec)
            log(rec)
            good_state = {"model": copy.deepcopy(model.state_dict()),
                          "opt": copy.deepcopy(opt.state_dict())}

    if cfg.out_dir:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
        (out / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    return history
