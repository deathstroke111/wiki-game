# Architecture decision records

Decisions made while building the Wikipedia-game RL pipeline (2026-11-06 to 2026-11-09), recorded retrospectively on 2026-11-10 from the session transcripts, the git history, the README and CLAUDE.md. Format: MADR (see `.claude/skills/adr-skill`). Cross-cutting takeaways are in [../LESSONS.md](../LESSONS.md).

**Reading note.** Several ADRs cite `CLAUDE.md` as provenance. It is no longer tracked ([ADR-0022](0022-stop-tracking-claude-md.md)); old versions are readable with `git show 9189b84:CLAUDE.md`. Where a rationale was reconstructed (the option list shown at the time was not preserved) the ADR says so. Numbers come from the W&B runs and episode files cited in each ADR.

| # | Decision | Status | Decided |
|---|---|---|---|
| [0001](0001-train-on-an-offline-link-graph.md) | Train and evaluate on an offline link graph | accepted | 2026-11-06 |
| [0002](0002-wikispeedia-first-dataset.md) | Wikispeedia first; Simple English Wikipedia deferred | accepted | 2026-11-06 |
| [0003](0003-environment-rules-and-action-format.md) | Environment rules and `<think>/<answer>` action format follow the paper | accepted | 2026-11-06 |
| [0004](0004-stack-and-two-machine-workflow.md) | PyTorch + TRL in a uv venv; Mac dev, rented GPU training, machine hook | accepted | 2026-11-06 |
| [0005](0005-heldout-evaluation-protocol.md) | One shared held-out evaluation protocol | accepted | 2026-11-06 |
| [0006](0006-streamlit-app-with-wandb-as-source-of-truth.md) | Streamlit app, W&B as the single source of truth | accepted | 2026-11-06 |
| [0007](0007-invalid-answer-causes-and-optional-feedback.md) | Classify invalid answers; opt-in feedback flag | accepted | 2026-11-07 |
| [0008](0008-baselines-at-50-pairs-per-hop.md) | Baselines at 50 pairs per hop on the Mac | accepted | 2026-11-07 |
| [0009](0009-switch-to-r1-distill-qwen-1-5b.md) | 1.5B model; switch Qwen2.5-Instruct to R1-Distill-Qwen-1.5B | accepted | 2026-11-07 |
| [0010](0010-per-step-progress-shaped-reward.md) | Per-step reward with progress shaping | accepted | 2026-11-06 |
| [0011](0011-ranked-invalid-rewards-no-warm-start.md) | Ranked invalid-answer rewards; no warm start | accepted (refines the earlier fixed -2 floor) | 2026-11-07 |
| [0012](0012-trl-grpo-with-lora-per-step.md) | TRL GRPOTrainer + LoRA | accepted | 2026-11-07 |
| [0013](0013-mixed-training-state-distribution.md) | Mixed training states (start / optimal / noisy) | accepted (known easy skew) | 2026-11-07 |
| [0014](0014-completion-cap-curriculum.md) | Completion-cap curriculum 1024 then 512 | accepted | 2026-11-07 |
| [0015](0015-tiny-model-plumbing-on-the-mac.md) | Tiny-model plumbing on the Mac, R1 on the GPU | accepted | 2026-11-07 |
| [0016](0016-windows-rtx-run-rules.md) | Operating rules from the Windows RTX box | accepted (Windows-specific) | 2026-11-08 |
| [0017](0017-unattended-chained-stages.md) | Unattended, gated stage chain | accepted | 2026-11-08 |
| [0018](0018-wandb-artifact-handoff.md) | Verified W&B-artifact handoff in two phases | accepted | 2026-11-08 |
| [0019](0019-working-agreements.md) | Working agreements owner / agents | accepted | 2026-11-06 |
| [0020](0020-small-eval-on-the-gpu-box.md) | Evaluate the four R1 checkpoints at 20 pairs per hop | accepted (wide CIs) | 2026-11-08 |
| [0021](0021-try-runpod-on-an-experiment-branch.md) | Try Runpod on an experiment branch | accepted (in use; owner to confirm; amended 2026-11-10) | 2026-11-09 |
| [0022](0022-stop-tracking-claude-md.md) | Stop tracking CLAUDE.md; agent notes stay local | accepted | 2026-11-10 |
| [0023](0023-stage-3-harder-training-states-on-runpod.md) | Stage 3: harder training states, on Runpod | proposed (run in progress) | 2026-11-09 |

## Relationships

* 0001 → 0002 (which graph); 0003 defines the action format that 0007 and 0011 classify and score.
* 0010 (shaped reward) → 0011 (how invalid answers score below it) → 0012 (trainer) → 0013 (states) → 0014 (cap curriculum).
* 0009 (model) motivated 0011's cold-start handling; 0015 explains how that problem was discovered.
* 0004/0016 (machines) → 0017/0018 (running and moving work) → 0020 (evals) → 0021 (next provider) → 0023 (first run there: harder states, following up the findings of 0013 and 0020).
* 0022 changes where agent guidance lives; it amends 0018 (git-polled status) and 0019 (the working agreements, now tracked only here).

## Conventions

* Numeric prefix, kebab-case slug, MADR front matter. Status is one of `proposed`, `accepted`, `rejected`, `deprecated`, `superseded by ADR-NNNN`.
* The "Implementation Plan" section in these retrospective ADRs lists where the decision lives in the repo and how to check it still holds.
* Add `ADR-NNNN` comments in code at entry points when changing an area an ADR governs; supersede with a new ADR rather than rewriting history.
* Open items that should become ADRs when decided: on-policy state rounds, trajectory-only reward ablation, Simple English Wikipedia graph, the outcome of stage 3 (0023), keeping durable agent rules in a tracked file (0022).
