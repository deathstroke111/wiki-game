---
status: "accepted"
date: 2026-11-07
decision-makers: "project owner (user)"
consulted: "dev agent (Claude), recommended option chosen"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Implement per-step GRPO with TRL's GRPOTrainer and a LoRA adapter

## Context and Problem Statement

We needed the training loop. Each training example is one game state (the prompt); the model samples G answers; each is scored by taking that move in the environment; the group-relative reward drives the update. TRL and PEFT were not yet in the project venv.

## Decision Drivers

* Fit a 1.5B model in 16 GB VRAM.
* Avoid writing and debugging a custom RL loop.
* Keep checkpoints small and easy to ship between machines ([ADR-0018](0018-wandb-artifact-handoff.md)).

## Considered Options

* TRL `GRPOTrainer` + LoRA (chosen, "Recommended").
* A custom hand-written GRPO loop. *Other alternatives shown in the dialog were not preserved.*

## Decision Outcome

Chosen option: "TRL GRPOTrainer + LoRA", with the defaults below. The reward function rebuilds each state with `env.set_state` and replays the sampled move (`grpo_reward.StepReward`).

* TRL 1.17.1, PEFT LoRA r=16, alpha 32, dropout 0.05, all linear layers; bf16 on CUDA, fp32 elsewhere; gradient checkpointing.
* G=8 answers per prompt, 4 prompts per optimizer step (32 sequences), micro-batch 4, T=0.7, lr 1e-5, `beta=0` (no reference model), TRL default loss `dapo`, group-normalised advantages.
* Dataset rows carry source, target, path and invalid count so the state is rebuilt exactly.

### Consequences

* Good, because it worked: stage 1b raised success and removed unparseable output; learning appeared without hyper-parameter changes (invalid 96% to 36% in 275 steps).
* Good, because adapters are ~70 MB each, so every checkpoint is shippable.
* Bad, because plain HF generation is slow: ~85-105 s per step at the 1024-token cap (~7.5-9 h for 300 steps); vLLM colocate mode is the upgrade path but is unavailable on Windows.
* Bad, because `--resume-adapter` restores weights only, not optimizer state.
* Neutral, because TRL's `log_completions` prints sampled text, which crashed the first Windows launch on a cp1252 console ([ADR-0016](0016-windows-rtx-run-rules.md)).

## Implementation Plan

* **Affected paths**: `scripts/grpo_train.py`, `src/wikigame/grpo_reward.py`, `src/wikigame/states.py`, `pyproject.toml` (`trl`, `peft` in group `llm`), `tests/test_grpo_parts.py`.
* **Patterns to follow**: `--micro-batch` must divide prompts-per-step x G; `--max-steps` also sizes the state dataset.

### Verification

- [ ] `uv run pytest tests/test_grpo_parts.py` passes.
- [ ] Plumbing check on the Mac: `HF_HOME=data/hf uv run python scripts/grpo_train.py --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 3 --num-generations 4 --prompts-per-step 1 --micro-batch 4 --max-completion 128 --no-wandb` completes (numbers meaningless, see [ADR-0015](0015-tiny-model-plumbing-on-the-mac.md)).

## More Information

Provenance: AskUserQuestion answer 2026-11-07 20:31, commit 140cf55. Recorded retrospectively on 2026-11-10.
