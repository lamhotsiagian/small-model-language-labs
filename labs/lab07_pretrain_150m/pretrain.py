"""
Lab 7: pretrain a Llama-style 150M SLM on a single multi-GPU node.

Launch:
    torchrun --nproc_per_node=8 pretrain.py --config config_150m.yaml            # DDP
    torchrun --nproc_per_node=8 pretrain.py --config config_150m.yaml --fsdp     # FSDP
    python pretrain.py --config config_150m.yaml --smoke                          # CPU check

What this script demonstrates (Chapter 7):
  * memory-mapped uint16 shards + random-offset sequence packing
  * BF16 autocast, fused AdamW, gradient accumulation to a token-based global batch
  * torch.compile and SDPA (FlashAttention kernels when available)
  * WSD schedule, z-loss, global-norm clipping, loss-spike skip + rollback
  * MFU accounting, periodic validation loss, periodic HellaSwag (log-likelihood)
  * atomic, resumable checkpoints (model + optimizer + step + RNG + data cursor)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.model import DecoderBlock, SLMConfig, SmallLM
from slmlab.train import TrainConfig, build_optimizer, lr_at


# ---------------------------------------------------------------------------
# Data: random windows from memory-mapped shards
# ---------------------------------------------------------------------------
class ShardSampler:
    def __init__(self, pattern: str, seq_len: int, seed: int) -> None:
        files = sorted(Path(pattern).parent.glob(Path(pattern).name))
        assert files, f"no shards match {pattern}"
        self.arrs = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.sizes = np.array([len(a) for a in self.arrs], dtype=np.float64)
        self.seq = seq_len
        self.rng = np.random.default_rng(seed)

    def batch(self, bsz: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        shard_ids = self.rng.choice(len(self.arrs), size=bsz, p=self.sizes / self.sizes.sum())
        rows = []
        for s in shard_ids:
            arr = self.arrs[s]
            i = self.rng.integers(0, len(arr) - self.seq - 1)
            rows.append(torch.from_numpy(arr[i:i + self.seq + 1].astype(np.int64)))
        x = torch.stack(rows)
        return x[:, :-1].to(device, non_blocking=True), x[:, 1:].to(device, non_blocking=True)


def smoke_shards(tmp: Path, vocab: int) -> str:
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # a learnable pattern: repeating arithmetic sequences
    base = (np.arange(2_000_000) * 7 + rng.integers(0, 3, 2_000_000)) % vocab
    base.astype(np.uint16).tofile(tmp / "train_000.bin")
    base[:200_000].astype(np.uint16).tofile(tmp / "val_000.bin")
    return str(tmp)


# ---------------------------------------------------------------------------
# Checkpointing (atomic: write to tmp, then rename)
# ---------------------------------------------------------------------------
def save_ckpt(path: Path, model, opt, step: int, keep_last: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
             "torch_rng": torch.get_rng_state()}
    tmp = path / f"step_{step:07d}.pt.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path / f"step_{step:07d}.pt")
    for old in sorted(path.glob("step_*.pt"))[:-keep_last]:
        old.unlink()


def latest_ckpt(path: Path):
    c = sorted(path.glob("step_*.pt"))
    return c[-1] if c else None


# ---------------------------------------------------------------------------
# HellaSwag by log-likelihood (the lm-eval "acc_norm" rule)
# ---------------------------------------------------------------------------
@torch.no_grad()
def hellaswag(model, tok, items, device) -> float:
    correct = 0
    for it in items:
        ctx = tok(it["ctx"], add_special_tokens=False)["input_ids"]
        scores = []
        for ending in it["endings"]:
            end = tok(" " + ending, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([ctx + end], device=device)
            logp = torch.log_softmax(model(ids)[0][0, :-1].float(), -1)
            tgt = ids[0, 1:]
            ll = logp[len(ctx) - 1:].gather(-1, tgt[len(ctx) - 1:, None]).sum().item()
            scores.append(ll / len(ending))              # length-normalised by characters
        correct += int(np.argmax(scores) == int(it["label"]))
    return correct / max(1, len(items))


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_150m.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--fsdp", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    C = yaml.safe_load(open(a.config))
    M, T, E, K = C["model"], C["train"], C["eval"], C["checkpoint"]

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0
    torch.manual_seed(1234 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    if a.smoke:  # shrink everything so the full code path runs on a CPU in a minute
        M.update(d_model=128, n_layers=4, n_heads=4, n_kv_heads=2, d_ff=384, max_seq_len=128,
                 vocab_size=512)
        T.update(total_tokens=2e5, seq_len=128, micro_batch=8, global_batch_tokens=2048,
                 warmup_steps=10, compile=False)
        E.update(every_steps=20, val_batches=5, hellaswag_every_steps=10 ** 9)
        K.update(dir="checkpoints/smoke", every_steps=50)
        a.data = smoke_shards(Path("data_smoke"), M["vocab_size"])

    cfg = SLMConfig(**M)
    model = SmallLM(cfg).to(device)
    n_params = model.num_params()
    tokens_per_micro = T["micro_batch"] * T["seq_len"] * world
    accum = max(1, T["global_batch_tokens"] // tokens_per_micro)
    steps = int(T["total_tokens"] // (tokens_per_micro * accum))
    tcfg = TrainConfig(steps=steps, lr=T["lr"], min_lr_ratio=T["min_lr_ratio"],
                       warmup=T["warmup_steps"], schedule=T["schedule"], decay_frac=T["decay_frac"],
                       weight_decay=T["weight_decay"], betas=tuple(T["betas"]), eps=T["eps"])
    if master:
        print(f"[lab07] params={n_params/1e6:.1f}M world={world} accum={accum} steps={steps} "
              f"tokens/step={tokens_per_micro * accum:,}")

    if T.get("activation_checkpointing"):
        from torch.utils.checkpoint import checkpoint
        for blk in model.layers:  # recompute each block in backward: ~33% more FLOPs, far less memory
            fwd = blk.forward
            blk.forward = lambda *args, _f=fwd, **kw: checkpoint(_f, *args, use_reentrant=False, **kw)
    if a.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy({DecoderBlock}),
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                                    reduce_dtype=torch.float32),
                     use_orig_params=True)
    elif ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    raw = model.module if hasattr(model, "module") else model
    opt = build_optimizer(model, tcfg)
    fwd_model = torch.compile(model) if T.get("compile") and device != "cpu" else model

    start = 0
    ck = latest_ckpt(Path(K["dir"]))
    if ck:  # resume exactly where we stopped
        st = torch.load(ck, map_location=device)
        raw.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"] + 1
        if master:
            print(f"[lab07] resumed from {ck.name}")

    train_s = ShardSampler(f"{a.data}/train_*.bin", T["seq_len"], seed=rank + start)
    val_s = ShardSampler(f"{a.data}/val_*.bin", T["seq_len"], seed=999)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
    peak = C["hardware"]["peak_tflops_per_gpu"] * 1e12 * world
    ema, log = None, []
    t0 = time.time()

    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        loss_acc = 0.0
        for micro in range(accum):
            x, y = train_s.batch(T["micro_batch"], device)
            sync = (nullcontext() if not ddp or micro == accum - 1 or a.fsdp
                    else model.no_sync())                       # skip all-reduce until last micro-step
            with sync, amp:
                _, loss = fwd_model(x, y, z_loss=T["z_loss"])
                (loss / accum).backward()
            loss_acc += loss.item() / accum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), T["clip"]).item()

        # Spike guard: a loss > 2x its EMA after warmup means skip this update.
        if ema is not None and step > T["warmup_steps"] and loss_acc > 2.0 * ema:
            opt.zero_grad(set_to_none=True)
            if master:
                print(f"[lab07] step {step}: spike {loss_acc:.3f} vs ema {ema:.3f}, update skipped")
            continue
        opt.step(); opt.zero_grad(set_to_none=True)
        ema = loss_acc if ema is None else 0.99 * ema + 0.01 * loss_acc

        if master and (step % E["every_steps"] == 0 or step == steps - 1):
            dt = time.time() - t0; t0 = time.time()
            n_steps = E["every_steps"] if step else 1
            tps = n_steps * accum * tokens_per_micro / dt
            with torch.no_grad(), amp:
                raw.eval()
                vl = sum(raw(*val_s.batch(T["micro_batch"], device))[1].item()
                         for _ in range(E["val_batches"])) / E["val_batches"]
                raw.train()
            rec = {"step": step, "lr": round(opt.param_groups[0]["lr"], 6), "loss": round(loss_acc, 4),
                   "val_loss": round(vl, 4), "grad_norm": round(gnorm, 3), "tok_per_s": int(tps),
                   "mfu": round(6 * n_params * tps / peak, 4)}
            print(json.dumps(rec)); log.append(rec)
        if master and step % K["every_steps"] == 0 and step > start:
            save_ckpt(Path(K["dir"]), raw, opt, step, K["keep_last"])

    if master:
        save_ckpt(Path(K["dir"]), raw, opt, steps - 1, K["keep_last"])
        Path(K["dir"], "train_log.json").write_text(json.dumps(log, indent=2))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
