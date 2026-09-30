"""
Lab 8: distillation losses (Chapter 8, sections 8.1-8.4).

All functions take student logits and teacher logits of shape [B, T, V]
(or a sparse top-k teacher) plus a mask [B, T] selecting the response tokens
that should be distilled (prompt tokens are excluded).

  forward_kl   KL(p_T || p_S)  "mean-seeking": student covers all teacher modes
  reverse_kl   KL(p_S || p_T)  "mode-seeking": student concentrates on teacher's
                               main modes; preferred for generation (Gu et al., 2024)
  gkd_jsd      generalised Jensen-Shannon with interpolation beta (Agarwal et al., 2024):
               beta -> 0 approaches forward KL, beta -> 1 approaches reverse KL
  topk_kl      forward KL against a teacher stored as top-k (ids, logprobs), the
               format offline distillation pipelines write to disk

Temperature tau softens both distributions; the tau^2 factor keeps gradient
magnitudes comparable across temperatures (Hinton et al., 2015).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1)


def forward_kl(s_logits, t_logits, mask, tau: float = 1.0):
    t_logp = F.log_softmax(t_logits.float() / tau, -1)
    s_logp = F.log_softmax(s_logits.float() / tau, -1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    return _masked_mean(kl, mask) * tau ** 2


def reverse_kl(s_logits, t_logits, mask, tau: float = 1.0):
    t_logp = F.log_softmax(t_logits.float() / tau, -1)
    s_logp = F.log_softmax(s_logits.float() / tau, -1)
    kl = (s_logp.exp() * (s_logp - t_logp)).sum(-1)
    return _masked_mean(kl, mask) * tau ** 2


def gkd_jsd(s_logits, t_logits, mask, beta: float = 0.5, tau: float = 1.0):
    """JSD_beta(p_T || p_S) with mixture m = beta * p_T + (1 - beta) * p_S."""
    t_logp = F.log_softmax(t_logits.float() / tau, -1)
    s_logp = F.log_softmax(s_logits.float() / tau, -1)
    if beta == 0.0:
        return forward_kl(s_logits, t_logits, mask, tau)
    if beta == 1.0:
        return reverse_kl(s_logits, t_logits, mask, tau)
    m_logp = torch.logsumexp(torch.stack([t_logp + math.log(beta),
                                          s_logp + math.log(1 - beta)]), dim=0)
    kl_t = (t_logp.exp() * (t_logp - m_logp)).sum(-1)
    kl_s = (s_logp.exp() * (s_logp - m_logp)).sum(-1)
    return _masked_mean(beta * kl_t + (1 - beta) * kl_s, mask) * tau ** 2


def topk_kl(s_logits, t_topk_ids, t_topk_logp, mask, tau: float = 1.0):
    """Forward KL restricted to the teacher's top-k tokens.

    The teacher's top-k probabilities are renormalised to sum to 1 (the tail is
    dropped). With k = 20-100 this keeps >95% of teacher mass for most tokens
    and cuts storage from V floats to 2k numbers per position.
    """
    t_logp = F.log_softmax(t_topk_logp.float() / tau, -1)          # renormalise over k
    s_logp_all = F.log_softmax(s_logits.float() / tau, -1)
    s_logp = s_logp_all.gather(-1, t_topk_ids)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    return _masked_mean(kl, mask) * tau ** 2


def sft_ce(s_logits, labels, mask):
    """Sequence-level KD: plain cross-entropy on teacher-generated tokens."""
    ce = F.cross_entropy(s_logits.float().transpose(1, 2), labels, reduction="none")
    return _masked_mean(ce, mask)


if __name__ == "__main__":
    # Mode-seeking vs mean-seeking. A capacity-limited student (one Gaussian bump
    # over a 32-token vocabulary) imitates a BIMODAL teacher, the way a small model
    # must approximate a richer teacher distribution. Multi-start optimisation
    # finds each divergence's best solution.
    V = 32
    pos = torch.arange(V).float()
    t_p = 0.55 * torch.exp(-(pos - 8) ** 2 / 8) + 0.45 * torch.exp(-(pos - 22) ** 2 / 8) + 1e-6
    teacher, mask = t_p.log().view(1, 1, V), torch.ones(1, 1)

    def fit(fn, mu0):
        mu = torch.tensor(mu0, requires_grad=True)
        log_sigma = torch.tensor(0.0, requires_grad=True)
        opt = torch.optim.Adam([mu, log_sigma], lr=0.05)
        for _ in range(3000):
            s_logits = (-(pos - mu) ** 2 / (2 * log_sigma.exp() ** 2)).view(1, 1, V)
            opt.zero_grad(); loss = fn(s_logits, teacher, mask); loss.backward(); opt.step()
        return loss.item(), mu.item(), log_sigma.exp().item(), s_logits.detach().softmax(-1)[0, 0]

    print("teacher: 55% mass near token 8, 45% near token 22")
    for name, fn in [("forward KL", forward_kl), ("reverse KL", reverse_kl),
                     ("GKD beta=0.9", lambda s, t, m: gkd_jsd(s, t, m, 0.9))]:
        loss, mu, sigma, p = min((fit(fn, m0) for m0 in (6.0, 10.0, 15.0, 20.0, 24.0)), key=lambda r: r[0])
        print(f"{name:<13} mu={mu:5.1f} sigma={sigma:4.1f}  mass@mode1={p[4:13].sum():.2f} "
              f"mass@mode2={p[18:27].sum():.2f} between={p[13:18].sum():.2f}")
