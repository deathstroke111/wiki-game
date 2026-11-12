"""The training loop: generate -> score -> advantages -> loss -> update -> hand the new LoRA to the sampler.

One step:
  1. the source returns ``prompts_per_step`` groups of G scored answers (``sources.StepSource``)
  2. advantages per group (``algorithms.advantages``); with ``drop_zero_std`` groups whose answers all scored the
     same are dropped (they carry no signal under a group baseline)
  3. the "old" log-probs (the policy that sampled) are recomputed by the trainer in one no-grad pass when they are
     needed: batch reuse (``updates_per_batch`` > 1) or a vLLM sampler (its log-probs differ slightly from the
     trainer's; logged as ``sampler/logprob_abs_diff``, optionally corrected with truncated importance sampling,
     ``tis_cap``). Otherwise old = current (detached), so the ratio is exactly 1, as in every TRL stage so far
  4. ``updates_per_batch`` optimizer updates over the batch in micro-batches (gradient accumulation with batch-wide
     normalisers, so micro-batching does not change the gradient)
  5. the sampler gets the new adapter (vLLM: saved to ``sync_dir`` and loaded as a fresh LoRA; HF: live weights)
  6. optional on-policy round, logging, checkpoint
Metric names follow TRL's where one exists (reward, reward_std, frac_reward_zero_std, entropy, clip_ratio/*,
completions/*, env/*, onpolicy/*), so W&B charts of old and new runs line up.
"""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

from .algorithms import Algorithm, advantages, aggregate, per_token_loss, zero_std_groups
from .learner import Learner
from .sources import Group, OnPolicyRounds, StepSource


@dataclass
class TrainConfig:
    out: Path
    max_steps: int = 300
    prompts_per_step: int = 4
    num_generations: int = 8
    micro_batch: int = 8
    max_completion: int = 512
    updates_per_batch: int = 1
    tis_cap: float = 0.0  # 0 = off; else cap on the trainer/sampler probability ratio (TRL's token_truncate)
    save_steps: int = 25
    sleep_sampler: bool = False  # vLLM sleep mode during the update (frees its KV cache for the trainer)
    log_completions: bool = True
    extra: dict = field(default_factory=dict)


class RLTrainer:
    def __init__(self, learner: Learner, generator, source: StepSource, algorithm: Algorithm, cfg: TrainConfig,
                 rounds: OnPolicyRounds | None = None, log_fn: Callable[[dict, int], None] | None = None,
                 sync_dir: str | Path | None = None):
        self.learner, self.gen, self.source, self.alg, self.cfg = learner, generator, source, algorithm, cfg
        self.rounds, self.log_fn = rounds, log_fn
        self.out = Path(cfg.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sync_dir = Path(sync_dir or tempfile.mkdtemp(prefix="wikigame-lora-"))
        self.step = 0

    # ---- one step ----------------------------------------------------------------------------------------------
    def train_step(self) -> dict[str, float]:
        cfg, L = self.cfg, self.learner
        t0 = time.time()
        groups, metrics = self.source.next_batch(cfg.prompts_per_step)
        rewards = torch.tensor([g.rewards for g in groups], dtype=torch.float32)
        adv = advantages(self.alg, rewards)
        zero = zero_std_groups(rewards)
        metrics.update(self._reward_metrics(groups, rewards, zero))
        keep = [i for i in range(len(groups)) if not (self.alg.drop_zero_std and zero[i])]
        seqs = [(groups[i].prompt_ids, s.token_ids, float(adv[i, j]), s.logprobs)
                for i in keep for j, s in enumerate(groups[i].samples)]
        if cfg.log_completions:
            self._log_completions(groups, adv)
        if self.cfg.sleep_sampler and hasattr(self.gen, "sleep"):
            self.gen.sleep()
        t1 = time.time()
        if seqs:
            metrics.update(self._update(seqs))
        else:
            metrics["skipped_update"] = 1.0
        metrics["time/train_s"] = time.time() - t1
        t2 = time.time()
        if getattr(self.gen, "wants_adapter_files", False):
            self.gen.set_adapter(str(L.save_adapter(self.sync_dir / "live")))
        metrics["time/sync_s"] = time.time() - t2
        metrics["time/step_s"] = time.time() - t0
        if torch.cuda.is_available():
            metrics["sys/cuda_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        return metrics

    def _update(self, seqs) -> dict[str, float]:
        cfg, L, alg = self.cfg, self.learner, self.alg
        mb = max(1, cfg.micro_batch)
        chunks = [seqs[i:i + mb] for i in range(0, len(seqs), mb)]
        total_tokens = sum(len(c) for _, c, _, _ in seqs)
        norm = dict(total_tokens=total_tokens, total_seqs=len(seqs), max_completion=cfg.max_completion)
        sampler_lp = all(lp is not None for *_, lp in seqs)
        # a separate no-grad pass for the "old" log-probs only when the batch is reused: with one update, old = the
        # training pass's own detached log-probs, so the ratio is exactly 1 (a second pass differs by LoRA dropout)
        old: list[torch.Tensor | None] = [None] * len(chunks)
        m: dict[str, float] = {}
        if cfg.updates_per_batch > 1:
            with torch.no_grad():
                for k, ch in enumerate(chunks):
                    old[k] = L.token_logprobs([p for p, *_ in ch], [c for _, c, *_ in ch])[0].detach()
        sums = {"loss": 0.0, "ratio_sum": 0.0, "clip_low_sum": 0.0, "clip_high_sum": 0.0, "entropy_sum": 0.0,
                "tis_clipped": 0.0, "lp_diff": 0.0}
        last = {}
        for it in range(cfg.updates_per_batch):
            for k, ch in enumerate(chunks):
                lp, mask, ent = L.token_logprobs([p for p, *_ in ch], [c for _, c, *_ in ch], with_entropy=it == 0)
                ref = old[k] if old[k] is not None else lp.detach()
                if it == 0 and sampler_lp:  # trainer vs sampler (vLLM) log-probs of the same tokens
                    s = _pad([x for *_, x in ch], lp.shape[1], lp.device)
                    sums["lp_diff"] += float(((ref - s).abs() * mask).sum())
                a = torch.tensor([x for _, _, x, _ in ch], device=lp.device)
                tok_loss, st = per_token_loss(alg, lp, ref, a, mask)
                if cfg.tis_cap and sampler_lp:
                    s = _pad([x for *_, x in ch], lp.shape[1], lp.device)
                    w = torch.exp(ref - s)
                    sums["tis_clipped"] += float(((w > cfg.tis_cap).float() * mask).sum())
                    tok_loss = tok_loss * w.clamp(max=cfg.tis_cap).detach()
                loss = aggregate(alg, tok_loss, mask, **norm)
                loss.backward()
                sums["loss"] += float(loss.detach())
                for key in ("ratio_sum", "clip_low_sum", "clip_high_sum"):
                    sums[key] += float(st.get(key, 0.0))
                if ent is not None:
                    sums["entropy_sum"] += float((ent * mask).sum())
            last = L.optimizer_step()
        n_tok = max(1, total_tokens * cfg.updates_per_batch)
        m.update({"loss": sums["loss"] / cfg.updates_per_batch, "entropy": sums["entropy_sum"] / max(1, total_tokens),
                  "ratio_mean": sums["ratio_sum"] / n_tok, "clip_ratio/low_mean": sums["clip_low_sum"] / n_tok,
                  "clip_ratio/high_mean": sums["clip_high_sum"] / n_tok,
                  "clip_ratio/region_mean": (sums["clip_low_sum"] + sums["clip_high_sum"]) / n_tok,
                  "grad_norm": last.get("grad_norm", 0.0), "learning_rate": last.get("lr", 0.0),
                  "train/sequences": len(seqs), "train/tokens": total_tokens})
        if sampler_lp:
            m["sampler/logprob_abs_diff"] = sums["lp_diff"] / max(1, total_tokens)
        if cfg.tis_cap and sampler_lp:
            m["sampler/tis_clipped_ratio"] = sums["tis_clipped"] / n_tok
        return m

    def _reward_metrics(self, groups: list[Group], rewards: torch.Tensor, zero: torch.Tensor) -> dict[str, float]:
        lens = [len(s.token_ids) for g in groups for s in g.samples]
        cut = [not s.finished for g in groups for s in g.samples]
        std = rewards.std(dim=1) if rewards.shape[1] > 1 else torch.zeros(len(groups))
        return {"reward": float(rewards.mean()), "reward_std": float(std.mean()),
                "frac_reward_zero_std": float(zero.float().mean()),
                "completions/mean_length": sum(lens) / len(lens), "completions/max_length": max(lens),
                "completions/clipped_ratio": sum(cut) / len(cut)}

    def _log_completions(self, groups: list[Group], adv: torch.Tensor) -> None:
        with (self.out / "completions.jsonl").open("a") as f:
            for i, g in enumerate(groups):
                for j, (s, text) in enumerate(zip(g.samples, g.texts)):
                    f.write(json.dumps({"step": self.step + 1, "source": g.row["source"], "target": g.row["target"],
                                        "path": g.row["path"], "kind": g.row["kind"], "n_invalid": g.row["n_invalid"],
                                        "reward": g.rewards[j], "advantage": float(adv[i, j]),
                                        "tokens": len(s.token_ids), "finished": s.finished, "text": text}) + "\n")

    # ---- the run -----------------------------------------------------------------------------------------------
    def run(self) -> dict[str, float]:
        cfg = self.cfg
        pending: dict[str, float] = {}
        if self.rounds is not None:
            pending = self.rounds.play(self.step)
        metrics: dict[str, float] = {}
        while self.step < cfg.max_steps:
            metrics = self.train_step()
            self.step += 1
            metrics.update(pending)
            pending = {}
            if self.rounds is not None and self.rounds.due(self.step) and self.step < cfg.max_steps:
                pending = self.rounds.play(self.step)  # logged with the next step, like the TRL callback
            print(f"step {self.step}/{cfg.max_steps} reward {metrics['reward']:.3f} "
                  f"invalid {metrics.get('env/invalid_rate', float('nan')):.3f} "
                  f"zero_std {metrics['frac_reward_zero_std']:.2f} len {metrics['completions/mean_length']:.0f} "
                  f"| gen {metrics['time/generate_s']:.1f}s train {metrics['time/train_s']:.1f}s "
                  f"sync {metrics['time/sync_s']:.1f}s step {metrics['time/step_s']:.1f}s"
                  + (f" | lp_diff {metrics['sampler/logprob_abs_diff']:.4f}" if "sampler/logprob_abs_diff" in metrics else "")
                  + (f" | ratio {metrics['ratio_mean']:.3f} clip {metrics['clip_ratio/region_mean']:.3f}"
                     if "ratio_mean" in metrics else ""), flush=True)
            with (self.out / "metrics.jsonl").open("a") as f:  # always on disk, W&B or not
                f.write(json.dumps({"step": self.step, **metrics}) + "\n")
            if self.log_fn:
                self.log_fn(metrics, self.step)
            if cfg.save_steps and self.step % cfg.save_steps == 0:
                self.learner.save_checkpoint(self.out / f"checkpoint-{self.step}",
                                             {"step": self.step, "config": asdict(cfg)})
        self.learner.save_adapter(self.out)
        return metrics


def _pad(rows: list[list[float]], width: int, device) -> torch.Tensor:
    return torch.tensor([list(r) + [0.0] * (width - len(r)) for r in rows], dtype=torch.float32, device=device)
