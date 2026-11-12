"""Policy-gradient algorithms as plug-ins: an advantage estimator plus a per-token objective and an aggregation.

Every algorithm here consumes the same batch: groups of G sampled answers to one prompt, a scalar reward per answer,
and per-token log-probabilities under the current policy (``logp``) and under the policy that sampled the answers
(``old_logp``). They differ only in three choices, so each one is a preset of ``Algorithm``:

  baseline / scale   what is subtracted from the reward, and whether the result is divided by the group's std
  objective          pg     -ratio * A                          (plain policy gradient; ratio = 1 on-policy)
                     clip   -min(ratio * A, clip(ratio) * A)    (PPO-style; zero gradient once a token's ratio leaves
                                                                 [1 - eps_low, 1 + eps_high] in the direction of A)
                     cispo  -sg(min(ratio, cispo_max)) * A * logp (the IS weight is capped and detached, so every
                                                                 token keeps a gradient)
  ratio_level        token, or sequence (GSPO: one ratio per answer, the geometric mean of its token ratios)
  agg                token_mean     sum over all tokens / number of tokens in the batch (DAPO; no length bias)
                     seq_mean       mean over answers of (sum over its tokens / its length) (original GRPO, GSPO)
                     seq_sum_const  sum over all tokens / (answers x max_completion) (classic REINFORCE / RLOO,
                                    i.e. sum of log-probs per answer, scaled by a constant like Dr. GRPO)

With one optimizer update per generated batch (``updates_per_batch=1``) the ratio is exactly 1, so pg, clip and
cispo give the same gradient and the presets differ only in baseline, scaling and aggregation. Clipping and the
CISPO cap start to matter when batches are reused (updates_per_batch > 1) or when the sampler's log-probabilities
differ from the trainer's (vLLM). No KL term: every run so far used beta = 0, and DPO (which needs a reference model
and pairs) is not in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

BASELINES = ("group_mean", "leave_one_out", "batch_mean", "none")
OBJECTIVES = ("pg", "clip", "cispo")
AGGS = ("token_mean", "seq_mean", "seq_sum_const")


@dataclass(frozen=True)
class Algorithm:
    name: str
    baseline: str = "group_mean"
    scale_by_std: bool = True
    objective: str = "clip"
    ratio_level: str = "token"
    eps_low: float = 0.2
    eps_high: float = 0.2
    cispo_max: float = 5.0
    agg: str = "token_mean"
    drop_zero_std: bool = False  # DAPO dynamic sampling, filter-only: groups whose answers all scored the same are dropped

    def __post_init__(self):
        if self.baseline not in BASELINES or self.objective not in OBJECTIVES or self.agg not in AGGS \
                or self.ratio_level not in ("token", "sequence"):
            raise ValueError(f"bad algorithm settings: {self}")

    def with_overrides(self, **kw) -> "Algorithm":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


ALGORITHMS: dict[str, Algorithm] = {
    # Williams (1992) with a batch-mean baseline; one answer's log-prob is the sum over its tokens
    "reinforce": Algorithm("reinforce", baseline="batch_mean", scale_by_std=False, objective="pg", agg="seq_sum_const"),
    # REINFORCE Leave-One-Out (Ahmadian et al. 2024): baseline = mean reward of the other G-1 answers to the prompt
    "rloo": Algorithm("rloo", baseline="leave_one_out", scale_by_std=False, objective="pg", agg="seq_sum_const"),
    # GRPO as in DeepSeekMath: group-normalised advantage, PPO clip 0.2, per-answer length normalisation
    "grpo": Algorithm("grpo", objective="clip", eps_low=0.2, eps_high=0.2, agg="seq_mean"),
    # DAPO: clip-higher (0.2 / 0.28), token-level loss, zero-std groups filtered out
    "dapo": Algorithm("dapo", objective="clip", eps_low=0.2, eps_high=0.28, agg="token_mean", drop_zero_std=True),
    # what stages 1-4 ran (TRL loss_type="dapo" defaults): DAPO's token-level loss with a symmetric 0.2 clip, no filter
    "trl-dapo": Algorithm("trl-dapo", objective="clip", eps_low=0.2, eps_high=0.2, agg="token_mean"),
    # CISPO (MiniMax-M1): clip the IS weight, not the update; cap 5 as TRL recommends (ScaleRL)
    "cispo": Algorithm("cispo", objective="cispo", cispo_max=5.0, agg="token_mean"),
    # GSPO (Qwen): sequence-level ratio with the paper's tight clip range
    "gspo": Algorithm("gspo", objective="clip", ratio_level="sequence", eps_low=3e-4, eps_high=4e-4, agg="seq_mean"),
}


def get_algorithm(name: str) -> Algorithm:
    try:
        return ALGORITHMS[name]
    except KeyError:
        raise ValueError(f"unknown algorithm {name!r}; choose from {sorted(ALGORITHMS)}") from None


def advantages(alg: Algorithm, rewards: torch.Tensor, std_eps: float = 1e-4) -> torch.Tensor:
    """rewards: (n_groups, G) -> advantages of the same shape."""
    r = rewards.float()
    if alg.baseline == "group_mean":
        a = r - r.mean(dim=1, keepdim=True)
    elif alg.baseline == "leave_one_out":
        g = r.shape[1]
        if g < 2:
            raise ValueError("leave-one-out needs at least 2 answers per group")
        a = r - (r.sum(dim=1, keepdim=True) - r) / (g - 1)
    elif alg.baseline == "batch_mean":
        a = r - r.mean()
    else:
        a = r.clone()
    if alg.scale_by_std:
        a = a / (r.std(dim=1, keepdim=True) + std_eps) if r.shape[1] > 1 else a
    return a


def zero_std_groups(rewards: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """(n_groups,) bool: groups whose answers all scored the same (no learning signal under a group baseline)."""
    return (rewards.max(dim=1).values - rewards.min(dim=1).values) <= tol


def per_token_loss(alg: Algorithm, logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor,
                   mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """logp/old_logp/mask: (B, T); adv: (B,). Returns the unaggregated loss (B, T) and diagnostics (sums over
    active tokens, so micro-batches can be added up)."""
    a = adv[:, None].to(logp.dtype)
    log_ratio = logp - old_logp
    if alg.ratio_level == "sequence":
        seq = (log_ratio * mask).sum(1) / mask.sum(1).clamp(min=1)
        # GSPO-token form: same value as the sequence ratio, gradient flows through every token's log-prob
        log_ratio = logp - logp.detach() + seq.detach()[:, None]
    ratio = torch.exp(log_ratio)
    stats = {"ratio_sum": (ratio.detach() * mask).sum()}
    if alg.objective == "pg":
        loss = -ratio * a
    elif alg.objective == "clip":
        clipped = ratio.clamp(1 - alg.eps_low, 1 + alg.eps_high)
        loss = -torch.min(ratio * a, clipped * a)
        low = (ratio < 1 - alg.eps_low) & (a < 0)
        high = (ratio > 1 + alg.eps_high) & (a > 0)
        stats["clip_low_sum"] = (low.float() * mask).sum()
        stats["clip_high_sum"] = (high.float() * mask).sum()
    else:  # cispo
        weight = ratio.clamp(max=alg.cispo_max).detach()
        loss = -weight * a * logp
        stats["clip_high_sum"] = (((ratio > alg.cispo_max) & (a > 0)).float() * mask).sum()
    return loss, stats


def aggregate(alg: Algorithm, loss: torch.Tensor, mask: torch.Tensor, *, total_tokens: int, total_seqs: int,
              max_completion: int) -> torch.Tensor:
    """Reduce a micro-batch's (B, T) loss with batch-wide constants, so the micro-batch losses add up to the loss of
    the whole batch (gradient accumulation gives the same gradient as one big batch)."""
    masked = loss * mask
    if alg.agg == "token_mean":
        return masked.sum() / max(1, total_tokens)
    if alg.agg == "seq_mean":
        return (masked.sum(1) / mask.sum(1).clamp(min=1)).sum() / max(1, total_seqs)
    return masked.sum() / max(1, total_seqs * max_completion)
