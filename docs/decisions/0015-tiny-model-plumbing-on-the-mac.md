---
status: "accepted"
date: 2026-11-07
decision-makers: "project owner (user)"
consulted: "dev agent (Claude), recommended option chosen"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Develop the training loop on the Mac with a tiny model; run R1 only on the GPU box

## Context and Problem Statement

R1 costs ~25 s per generation on MPS, so even a few GRPO steps with it on the 24 GB Mac are impractical. But the training loop (dataset, reward replay, TRL integration, W&B logging, adapter save/load) has many places to break, and the rented GPU is "not always available".

## Decision Drivers

* Find plumbing bugs without paying for GPU time.
* Only pay GPU time for numbers that mean something.

## Considered Options

* Debug the loop directly with R1 on the Mac (too slow).
* Debug with a tiny model on the Mac, then run R1 on the RTX (chosen, "Recommended").
* Debug only on the rented GPU (wasteful, availability risk).

## Decision Outcome

Chosen option: "tiny model on the Mac for plumbing, R1 on CUDA". `grpo_train.py` accepts any `--model` and works with Qwen 0.5B/1.5B in fp32 on MPS.

### Consequences

* Good, because the Mac runs verified that the loop runs, gradients flow, W&B (offline) receives `env/*` metrics and completion tables, and adapters reload for evaluation, before any GPU time was spent.
* Bad, because plumbing numbers do not transfer: the Qwen 0.5B smoke run produced all-unparseable answers, so reward std was 0 and loss/grad_norm were 0, a *false alarm about the algorithm* that was really a model limitation. A 1.5B smoke run showed non-zero `grad_norm`. This is what motivated the cold-start work in [ADR-0011](0011-ranked-invalid-rewards-no-warm-start.md).
* Bad, because GPU memory and speed for R1 were unmeasured until the dry run on the RTX (8.6 GB peak, ~85-105 s per step).

## Implementation Plan

* **Affected paths**: `scripts/grpo_train.py`, `scripts/smoke_run.py`, `tests/test_grpo_parts.py`.
* **Command (Mac)**: `HF_HOME=data/hf uv run python scripts/grpo_train.py --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 3 --num-generations 4 --prompts-per-step 1 --micro-batch 4 --max-completion 128 --no-wandb`.
* **Patterns to follow**: choose a smoke model that yields reward variance; a smoke run with std 0 proves nothing about gradients.

### Verification

- [ ] The command above completes on the Mac and writes an adapter plus `wikigame_config.json`.
- [ ] Before any long GPU run, a 5-step dry run on the GPU reports peak VRAM, `frac_reward_zero_std`, cutoff rate and step time (done: 8.6 GB, 0, 44-50%, ~85-105 s).

## More Information

Provenance: AskUserQuestion answer 2026-11-07 20:31 and the smoke-run debugging that followed. Recorded retrospectively on 2026-11-10.
