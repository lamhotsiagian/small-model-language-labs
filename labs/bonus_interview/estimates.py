"""
Back-of-the-envelope estimates for the 20 system design interview cases
(Bonus chapter). Every number quoted in Step 2 ("Estimate the Scale") of each
case is printed by this script, so an interviewer's follow-up ("where did
that number come from?") always has an answer.

Run:  python labs/bonus_interview/estimates.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from slmlab.budget import ZOO, ArchSpec, BYTES, kv_bytes_per_token, param_breakdown  # noqa: E402

GB = 1e9
H100_BF16 = 989e12          # dense BF16 peak FLOP/s
H100_BW = 3.35e12           # HBM3 bytes/s
PHONE_BW = 60e9             # flagship LPDDR5X, sustained-ish bytes/s
LAPTOP_BW = 120e9           # Apple M-series (base/pro class) bytes/s


def hdr(n: int, title: str) -> None:
    print(f"\n=== Case {n}: {title}")


def params(a: ArchSpec) -> int:
    return param_breakdown(a)["total"]


# ---------------------------------------------------------------------------
def case01() -> None:
    hdr(1, "SLM-or-LLM decision platform")
    req_day, tin, tout = 50e6, 800, 200
    tok_day = req_day * (tin + tout)
    qps = req_day / 86400
    print(f"tokens/day = {tok_day/1e9:.0f}B, mean QPS = {qps:.0f}, peak (3x) = {3*qps:.0f}")
    # Illustrative blended prices per 1M tokens (assumptions, not quotes).
    for name, price in (("frontier API", 3.00), ("self-hosted 8B", 0.20), ("self-hosted 1B", 0.05)):
        print(f"  {name:16s} ${price:.2f}/M tok -> ${tok_day/1e6*price*30:,.0f}/month")
    share_small = 0.7
    blended = share_small * 0.05 + (1 - share_small) * 3.00
    print(f"  70% routed to 1B, 30% to API -> ${tok_day/1e6*blended*30:,.0f}/month")


def chinchilla_loss(n: float, d: float) -> float:
    # Hoffmann et al. (2022), Approach 3 fit.
    return 1.69 + 406.4 / n**0.34 + 410.7 / d**0.28


def case02() -> None:
    hdr(2, "Model size under a training budget plus inference demand")
    c = 1e21
    n_opt = math.sqrt(c / 120)
    d_opt = 20 * n_opt
    target = chinchilla_loss(n_opt, d_opt)
    d_inf = 10e9 * 24                            # 10B tokens/month for 24 months
    gpu_h = c / (H100_BF16 * 0.40) / 3600
    print(f"Chinchilla: N={n_opt/1e9:.2f}B, D={d_opt/1e9:.0f}B, loss={target:.4f}, "
          f"train = {gpu_h:,.0f} H100-hours at 40% MFU")
    best = None
    for n in [x * 1e8 for x in range(5, 60)]:
        # Solve for D that reaches the same loss at this N.
        rem = target - 1.69 - 406.4 / n**0.34
        if rem <= 0:
            continue
        d = (410.7 / rem) ** (1 / 0.28)
        total = 6 * n * d + 2 * n * d_inf
        if best is None or total < best[0]:
            best = (total, n, d)
    tot_chin = 6 * n_opt * d_opt + 2 * n_opt * d_inf
    total, n, d = best
    print(f"Inference-aware: N={n/1e9:.1f}B, D={d/1e9:.0f}B ({d/n:.0f} tok/param); "
          f"lifetime FLOPs {total:.2e} vs Chinchilla {tot_chin:.2e} "
          f"({(1-total/tot_chin)*100:.0f}% less)")


def case03() -> None:
    hdr(3, "1B architecture for phones")
    deep = ArchSpec("deep-thin-1B", 1536, 32, 24, 6, 4096, 49_152)
    wide = ZOO["Llama-3.2-1B"]
    for a in (deep, wide):
        p = params(a)
        w = p * BYTES["q4_k_m"]
        kv = kv_bytes_per_token(a, "fp16")
        emb = a.vocab * a.d_model / p
        print(f"{a.name:14s} L={a.n_layers:2d} d={a.d_model} params={p/1e9:.2f}B "
              f"embed share={emb*100:.0f}% Q4_K_M={w/GB:.2f}GB "
              f"KV={kv/1024:.0f}KiB/tok decode bound={PHONE_BW/w:.0f} tok/s")


def case04() -> None:
    hdr(4, "70B teacher -> 1B student")
    tokens = 20e9
    gen = 2 * 70e9 * tokens
    print(f"teacher forward over 20B tokens = {gen:.2e} FLOPs = "
          f"{gen/(H100_BF16*0.35)/3600:,.0f} H100-hours at 35% util")
    for k in (1, 32):
        b = k * 6 * tokens                         # int32 index + fp16 value
        print(f"  store top-{k} logits: {b/1e12:.2f} TB")
    print(f"  full 128K-vocab fp16 logits: {128_256*2*tokens/1e15:.1f} PB (infeasible)")
    student = 6 * 1.2e9 * tokens
    print(f"student training 20B tokens = {student:.2e} FLOPs "
          f"({student/(H100_BF16*0.4)/3600:,.0f} H100-hours)")


def case05() -> None:
    hdr(5, "Prune 8B -> 4B")
    scratch = 6 * 4.5e9 * 15e12
    prune = 6 * 4.5e9 * 94e9
    print(f"train 4.5B from scratch on 15T: {scratch:.2e} FLOPs; "
          f"prune+distill on 94B: {prune:.2e} FLOPs ({scratch/prune:.0f}x less)")
    imp = 2 * 8e9 * 1024 * 1024       # importance pass: 1024 samples x 1024 tokens
    print(f"activation-importance pass (1M tokens fwd on 8B): {imp:.2e} FLOPs "
          f"({imp/(H100_BF16*0.3):.0f} GPU-seconds)")


def case06() -> None:
    hdr(6, "Pretraining data pipeline")
    tokens = 1e12
    clean_tb = tokens * 4 / 1e12
    extracted_tb = clean_tb / 0.10
    docs = 2.0e9
    sig = docs * 112 * 4
    print(f"1T tokens ~ {clean_tb:.0f} TB clean text; at 10% retention need "
          f"~{extracted_tb:.0f} TB extracted text")
    print(f"MinHash signatures for {docs/1e9:.0f}B docs x 112 perms: {sig/1e12:.2f} TB")
    cls = 2 * 150e6 * 1000 * docs              # 150M classifier on first 1K tokens
    print(f"quality classifier (150M, 1K tokens/doc): {cls:.2e} FLOPs = "
          f"{cls/(H100_BF16*0.3)/3600:,.0f} H100-hours")


def case07() -> None:
    hdr(7, "Domain specialist")
    cpt = 6 * 1.24e9 * 2e9
    sft = 6 * 1.24e9 * 50e3 * 800
    print(f"CPT 2B tokens on 1.2B: {cpt:.2e} FLOPs = {cpt/(H100_BF16*0.4)/3600:.1f} H100-h; "
          f"SFT 50K x 800 tok: {sft:.2e} FLOPs")
    print("replay mix: 2B domain + 0.5B general (20%) tokens")


def lora_params(a: ArchSpec, r: int) -> int:
    d, kvd, f = a.d_model, a.n_kv_heads * a.hd, a.d_ff
    qd = a.n_heads * a.hd
    per = r * ((d + qd) + 2 * (d + kvd) + (qd + d) + 2 * (d + f) + (f + d))
    return per * a.n_layers


def case08() -> None:
    hdr(8, "200 adapters on one pool")
    a = ZOO["Llama-3.2-3B"]
    lp = lora_params(a, 16)
    base = params(a) * 2
    print(f"base 3B BF16 = {base/GB:.1f} GB; LoRA r=16 all-linear = {lp/1e6:.1f}M params "
          f"= {lp*2/1e6:.0f} MB; 200 adapters = {200*lp*2/GB:.1f} GB")
    kv = kv_bytes_per_token(a, "bf16")
    free = 80 * GB * 0.9 - base - 200 * lp * 2
    print(f"KV = {kv/1024:.0f} KiB/token; remaining HBM on 80GB (90% usable) = {free/GB:.0f} GB "
          f"= {free/kv/1e3:,.0f}K cached tokens")
    print(f"merged copies instead: 200 x {base/GB:.1f} GB = {200*base/GB:,.0f} GB")


def case09() -> None:
    hdr(9, "Reliable tool calling on 1B")
    tools, per = 20, 150
    print(f"schema prefix = {tools*per} tokens; at 5M calls/day prefix prefill = "
          f"{tools*per*5e6/1e9:.0f}B tokens/day without prefix caching")
    print(f"irrelevance: if 15% of turns need no tool and the model calls one 20% of "
          f"the time -> {0.15*0.2*100:.0f}% of all turns trigger a spurious action")


def case10() -> None:
    hdr(10, "Math reasoning on a budget")
    n, g, prompts, toks = 1.5e9, 8, 512, 1024
    gen = g * prompts * toks
    fl = 2 * n * gen + 6 * n * gen
    print(f"GRPO step: {gen/1e6:.1f}M generated tokens; ~{fl:.2e} FLOPs/step "
          f"(generation dominates wall-clock because decode is bandwidth bound)")
    print(f"500 steps = {500*gen/1e9:.1f}B generated tokens")
    print(f"test-time: 16 samples x 600 tokens = {16*600} tokens/question vs 600 greedy")


def case11() -> None:
    hdr(11, "Quantization choice")
    a = ZOO["Llama-3.2-3B"]
    p = params(a)
    for dt in ("bf16", "fp8", "q8_0", "q5_k_m", "q4_k_m", "q3_k_m"):
        w = p * BYTES[dt]
        print(f"  {dt:7s} {w/GB:5.2f} GB  phone bound {PHONE_BW/w:5.1f} tok/s  "
              f"H100 batch-1 bound {H100_BW/w:6.0f} tok/s")


def case12() -> None:
    hdr(12, "When on-device makes sense")
    for name in ("SmolLM2-360M", "Qwen2.5-0.5B", "Llama-3.2-1B", "Llama-3.2-3B"):
        a = ZOO[name]
        w = params(a) * BYTES["q4_k_m"]
        print(f"  {name:13s} Q4_K_M {w/GB:4.2f} GB  phone {PHONE_BW/w:5.0f} tok/s  "
              f"laptop {LAPTOP_BW/w:5.0f} tok/s")
    print(f"cloud offload: 10M users x 20 req/day = {10e6*20/86400:,.0f} req/s avoided")


def case13() -> None:
    hdr(13, "3B + 32K context on an 8GB phone")
    a = ZOO["Llama-3.2-3B"]
    w = params(a) * BYTES["q4_k_m"]
    for dt, name in (("bf16", "FP16 KV"), ("q8_0", "8-bit KV"), ("int4", "4-bit KV")):
        kv = kv_bytes_per_token(a, dt) * 32768
        print(f"  weights {w/GB:.2f} GB + {name} {kv/GB:.2f} GB + runtime 0.3 GB = "
              f"{(w+kv)/GB+0.3:.2f} GB")
    kvw = kv_bytes_per_token(a, "fp16") * 4096
    print(f"  sliding window 4K FP16 KV = {kvw/GB:.2f} GB")


def case14() -> None:
    hdr(14, "Hybrid on-device / cloud")
    req = 10e6 * 20
    for local in (0.70, 0.85, 0.95):
        cloud = req * (1 - local) / 86400
        print(f"  {local*100:.0f}% on device -> cloud mean {cloud:,.0f} req/s, peak(3x) {3*cloud:,.0f}")


def case15() -> None:
    hdr(15, "Make serving cheaper")
    base = 1.0
    small, large = 0.05, 1.0
    for acc in (0.5, 0.7, 0.85):
        cascade = small + (1 - acc) * large
        print(f"  cascade, small answers {acc*100:.0f}%: cost {cascade/base*100:.0f}% of baseline")
    hit = 0.2
    print(f"  + {hit*100:.0f}% semantic-cache hits on top of 70% cascade: "
          f"{(1-hit)*(small+0.3*large)*100:.0f}%")


def case16() -> None:
    hdr(16, "High-throughput serving capacity")
    a = ZOO["Llama-3.2-3B"]
    w = params(a) * 2
    kv = kv_bytes_per_token(a, "bf16")
    peak_rps, tin, tout = 400, 500, 250
    out_tps = peak_rps * tout
    batch = 128
    ctx = tin + tout / 2
    step = (w + batch * ctx * kv) / H100_BW
    ideal = batch / step
    eff = 0.6 * ideal
    gpus = out_tps / eff
    print(f"peak output = {out_tps:,.0f} tok/s; batch {batch} step >= {step*1e3:.1f} ms "
          f"-> {ideal:,.0f} tok/s ideal, {eff:,.0f} at 60% -> {gpus:.1f} H100 for decode")
    pre = peak_rps * tin * 2 * params(a)
    print(f"prefill = {pre:.2e} FLOP/s = {pre/(H100_BF16*0.5):.1f} H100 at 50% MFU")
    print(f"TPOT at batch {batch}: >= {step*1e3:.1f} ms  (Little: concurrency = "
          f"{peak_rps*tout*step*1.0/0.6:,.0f} sequences in flight)")


def case17() -> None:
    hdr(17, "Speculative decoding")
    for alpha in (0.6, 0.7, 0.8, 0.9):
        for gamma in (3, 5):
            e = (1 - alpha ** (gamma + 1)) / (1 - alpha)
            c = 0.1
            print(f"  alpha={alpha} gamma={gamma}: {e:.2f} tokens/target step, "
                  f"speedup ~{e/(gamma*c+1):.2f}x (draft cost ratio {c})")


def case18() -> None:
    hdr(18, "Prove 1B beats 7B")
    for p, half in ((0.7, 0.02), (0.7, 0.03)):
        n = 1.96**2 * p * (1 - p) / half**2
        print(f"  unpaired accuracy p={p}, +/-{half*100:.0f} pt 95% CI -> n = {n:,.0f} items")
    disc = 0.25
    n = 1.96**2 * disc / 0.03**2
    print(f"  paired, discordance {disc*100:.0f}%, +/-3 pt on the difference -> n ~ {n:,.0f}")


def case19() -> None:
    hdr(19, "Guardrail SLM")
    qps, toks = 2000, 1500
    for n in (1.0e9, 0.3e9):
        fl = 2 * n * qps * toks
        print(f"  {n/1e9:.1f}B classifier scanning {toks} tok/request @ {qps} QPS: "
              f"{fl:.2e} FLOP/s = {fl/(H100_BF16*0.4):.1f} H100 at 40% MFU")
    print(f"  scan only untrusted spans (~500 tok): "
          f"{2*0.3e9*qps*500/(H100_BF16*0.4):.1f} H100")


def case20() -> None:
    hdr(20, "MLOps for a fleet")
    models, targets = 6, 4
    print(f"artifacts per release = {models*targets}")
    full, delta = 1.9, 0.08
    users = 10e6
    print(f"OTA to {users/1e6:.0f}M devices: full {full*users/1e6:,.1f} PB vs "
          f"LoRA/delta {delta*users/1e6:,.1f} PB")


if __name__ == "__main__":
    for fn in (case01, case02, case03, case04, case05, case06, case07, case08, case09, case10,
               case11, case12, case13, case14, case15, case16, case17, case18, case19, case20):
        fn()
