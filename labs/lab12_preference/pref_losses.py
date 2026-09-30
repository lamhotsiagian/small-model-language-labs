"""
Lab 12: preference-optimisation objectives written out explicitly (Chapter 12).

Inputs are summed log-probabilities of the chosen (w) and rejected (l)
responses under the policy (pi_*) and, where needed, a frozen reference
(ref_*), plus response lengths. All losses return a scalar mean over pairs.

  dpo     -log sigma( beta * [(pi_w - ref_w) - (pi_l - ref_l)] )         Rafailov et al. (2023)
  ipo     ( h - 1/(2 tau) )^2 with h the same log-ratio margin (no beta)  Azar et al. (2024)
  simpo   -log sigma( beta/|w| * pi_w - beta/|l| * pi_l - gamma )         Meng et al. (2024)
          reference-free, length-normalised, target margin gamma
  orpo    NLL(w) - lambda * log sigma( log odds_w - log odds_l )          Hong et al. (2024)
          odds(y) = p/(1-p) with p = exp(mean token logprob); no reference
  kto     unpaired: desirable / undesirable examples vs a KL baseline     Ethayarajh et al. (2024)

The __main__ demo shows the property that matters most for small models:
how each loss behaves when the "preference" is really a LENGTH difference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo(pi_w, pi_l, ref_w, ref_l, beta: float = 0.1):
    margin = (pi_w - ref_w) - (pi_l - ref_l)
    return -F.logsigmoid(beta * margin).mean()


def ipo(pi_w, pi_l, ref_w, ref_l, tau: float = 0.1):
    h = (pi_w - ref_w) - (pi_l - ref_l)
    return ((h - 1 / (2 * tau)) ** 2).mean()


def simpo(pi_w, pi_l, len_w, len_l, beta: float = 2.0, gamma: float = 1.0):
    return -F.logsigmoid(beta * pi_w / len_w - beta * pi_l / len_l - gamma).mean()


def orpo(pi_w, pi_l, len_w, len_l, lam: float = 0.1):
    lp_w, lp_l = pi_w / len_w, pi_l / len_l                     # mean token log-prob
    log_odds = lambda lp: lp - torch.log1p(-torch.exp(lp).clamp(max=1 - 1e-6))
    ratio = -F.logsigmoid(log_odds(lp_w) - log_odds(lp_l))
    nll = -lp_w
    return (nll + lam * ratio).mean()


def kto(pi, ref, desirable: torch.Tensor, kl_ref: float, beta: float = 0.1,
        w_d: float = 1.0, w_u: float = 1.0):
    """desirable: bool tensor. kl_ref: batch estimate of KL(pi || ref) (detached)."""
    r = pi - ref
    v = torch.where(desirable, w_d * torch.sigmoid(beta * (r - kl_ref)),
                    w_u * torch.sigmoid(beta * (kl_ref - r)))
    lam = torch.where(desirable, torch.tensor(w_d), torch.tensor(w_u))
    return (lam - v).mean()


if __name__ == "__main__":
    # A pair where chosen is simply LONGER: same per-token quality (-1.2 nats/token),
    # 300 tokens vs 100 tokens. Summed log-probs therefore favour the SHORT one.
    per_tok = -1.2
    len_w, len_l = torch.tensor([300.0]), torch.tensor([100.0])
    pi_w, pi_l = per_tok * len_w, per_tok * len_l
    ref_w, ref_l = pi_w.clone(), pi_l.clone()                   # policy == reference at start
    print("same per-token quality, chosen is 3x longer; policy == reference:")
    print(f"  DPO  loss {dpo(pi_w, pi_l, ref_w, ref_l).item():.4f}  (= ln 2: no signal yet, margin 0)")
    print(f"  IPO  loss {ipo(pi_w, pi_l, ref_w, ref_l).item():.4f}")
    print(f"  SimPO loss {simpo(pi_w, pi_l, len_w, len_l).item():.4f}  (length-normalised: margin -gamma)")
    print(f"  ORPO loss {orpo(pi_w, pi_l, len_w, len_l).item():.4f}")
    # Gradient direction for DPO w.r.t. the chosen log-prob: DPO rewards raising
    # pi_w relative to ref. One cheap way to raise a SUMMED log-prob ratio is to
    # change length-sensitive behaviour, which is why DPO can drift verbose.
    pw = pi_w.clone().requires_grad_(True)
    dpo(pw, pi_l, ref_w, ref_l).backward()
    print(f"  dDPO/d(pi_w) = {pw.grad.item():+.4f}  (pushes chosen log-prob up, whatever its length)")
