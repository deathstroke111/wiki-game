# Lessons from building the Wikipedia-game RL pipeline

Cross-cutting reading of the [decision records](decisions/README.md) plus the run history (2026-11-06 to 2026-11-09). Evidence is cited by ADR and W&B run id. Where something is a judgement rather than a measurement it is labelled as such.

## Outcome in one paragraph

GRPO on a 1.5B R1 distill, with per-step shaped reward and ranked invalid-answer penalties, took the held-out success rate from **13.8% to 35.0%** (n=20 pairs per hop, 80 games) and cut invalid actions from 73% to 37%, but hop 4 stayed at 0% and the result is roughly on par with the Mac's **Qwen2.5-1.5B zero-shot (38.0%, n=50 per hop)**. So the pipeline works end to end and training clearly helps the *chosen* model, but the project has not yet shown that RL beat simply choosing a better base model ([ADR-0009](decisions/0009-switch-to-r1-distill-qwen-1-5b.md), [ADR-0020](decisions/0020-small-eval-on-the-gpu-box.md)).

## What worked (keep doing)

1. **Build evaluation and baselines before training.** The 50-per-hop Qwen baselines showed that ~60% of actions were *invalid*, not wrong navigation, and that naming an unlinked target was the dominant error (40-47% of actions). That reframed the whole project around validity and format ([ADR-0007](decisions/0007-invalid-answer-causes-and-optional-feedback.md), [ADR-0008](decisions/0008-baselines-at-50-pairs-per-hop.md)).
2. **Make failure causes first-class data.** A cause label on every invalid action cost a few hours and paid for itself in diagnosis, in the app, and in reward design.
3. **Design the reward for the cold start.** Untrained R1 was 94-97% invalid; a flat penalty would have given zero-variance groups. Ranked tiers let learning start immediately: invalid 96% to 36% in 275 steps with default hyper-parameters ([ADR-0011](decisions/0011-ranked-invalid-rewards-no-warm-start.md)).
4. **Per-step shaped reward gives signal from step 1** and keeps eval honest because eval stays unshaped ([ADR-0010](decisions/0010-per-step-progress-shaped-reward.md)).
5. **Cheap curriculum, large speed-up.** The 512-token second stage ran at ~40-55 s per step versus ~85 s and reached the best result ([ADR-0014](decisions/0014-completion-cap-curriculum.md)).
6. **Opt-in flags (default off) keep old results comparable**: feedback, shaping, lenient parsing ([ADR-0003](decisions/0003-environment-rules-and-action-format.md), [ADR-0007](decisions/0007-invalid-answer-causes-and-optional-feedback.md)).
7. **Checkpoint everything and ship it verifiably.** Keeping every adapter and moving them via sha256-checked W&B artifacts meant any checkpoint can be evaluated anywhere ([ADR-0016](decisions/0016-windows-rtx-run-rules.md), [ADR-0018](decisions/0018-wandb-artifact-handoff.md)).
8. **Written handoffs in the repo** (CLAUDE.md with STATUS lines) let two agents on two machines and several sessions stay in sync ([ADR-0019](decisions/0019-working-agreements.md)).

## What cost time, and the rule it implies

| What happened | Cost | Rule |
|---|---|---|
| Second GPU job during training made Windows spill VRAM into system RAM; steps 85 s to 380-480 s, GPU "99%" but ~87 W | ~3 h, first run killed at step 32 | In any new GPU environment, check power draw and step time in the first steps; never run two GPU jobs ([ADR-0016](decisions/0016-windows-rtx-run-rules.md)) |
| Windows console cp1252 crashed on TRL's logged completions | one failed launch | Force UTF-8; the dry run (with W&B off) could not hit the bug, so dry-run with the real logging on |
| The "CUDA box" was Windows; the Linux-only hook branch had never run | rework | Code paths written on one machine for another are untested until run there ([ADR-0004](decisions/0004-stack-and-two-machine-workflow.md)) |
| `.gitignore` line `data/` also ignored `src/wikigame/data/` | loader missing from GitHub for a day; found only because the owner asked | Anchor ignore rules (`/data/`) and check a fresh clone after changing them ([ADR-0002](decisions/0002-wikispeedia-first-dataset.md)) |
| `uv sync --group llm --group dev` silently uninstalled streamlit | app tests failed after the transfer | Always list every group you still need ([ADR-0004](decisions/0004-stack-and-two-machine-workflow.md)) |
| zsh does not word-split unquoted `$var`: `"--lenient --feedback"` became one argument | one failed baseline run | Pass flags as separate words |
| 10 pairs per hop gave about +-30-point intervals; early numbers were too optimistic | runs repeated | Fix the evaluation size before drawing conclusions ([ADR-0008](decisions/0008-baselines-at-50-pairs-per-hop.md)) |
| Tiny-model smoke run had reward std 0, so loss and grad_norm were 0 | confusion about whether the algorithm worked | A smoke test must produce reward variance to prove gradients flow ([ADR-0015](decisions/0015-tiny-model-plumbing-on-the-mac.md)) |

## Judgement calls that deserve a second look

* **Switching models on a hypothesis.** The move from Qwen to R1 was motivated by "Qwen is bad at instruction following". It was never measured against an R1 baseline, which turned out to be much *worse* than Qwen (13.8% vs 38%). About 13 GPU-hours of training brought R1 to parity. A 20-minute untrained baseline of the candidate model would have set honest expectations. This was a deliberate owner choice ("see how it performs after training"); the lesson is about cost, not blame.
* **Train/eval mismatch.** 57% of training states were one move from the target, so per-step training success (47%) overstated game success (35%) and no hop-4 game was solved ([ADR-0013](decisions/0013-mixed-training-state-distribution.md)).
* **No ablations were run**, so we cannot attribute gains: ranked vs flat invalid rewards, shaped vs trajectory-only reward, curriculum vs 512 from the start, and feedback in training are all unmeasured ([ADR-0010](decisions/0010-per-step-progress-shaped-reward.md), [ADR-0011](decisions/0011-ranked-invalid-rewards-no-warm-start.md), [ADR-0014](decisions/0014-completion-cap-curriculum.md)).
* **Small evals on the GPU box** (n=20 per hop) cannot separate step 100 from the final model or the trained R1 from Qwen ([ADR-0020](decisions/0020-small-eval-on-the-gpu-box.md)).
* **Unreproducible launchers.** The chain scripts were gitignored `.ps1` files ([ADR-0017](decisions/0017-unattended-chained-stages.md)).

## Process lessons

* The consult-first, options-with-trade-offs workflow ([ADR-0019](decisions/0019-working-agreements.md)) produced decisions the owner understood and could reverse; its cost was speed, and CLAUDE.md grew long (it needed condensing once). Move settled decisions into ADRs and keep CLAUDE.md to current state and handoff.
* Polling for a handoff every 20 minutes works but is slow and dies with the session; a pushed notification or a longer sleep would be cheaper.
* Never delete experiment records without asking: the one record of the slowdown lives in a run that looks "crashed".

## Suggested next steps (ordered)

1. Run the deferred **R1 baseline at 50 pairs per hop**, plus evals of stage 2 step 100 and final at the same n, on the new GPU provider ([ADR-0021](decisions/0021-try-runpod-on-an-experiment-branch.md)). Decide from intervals, not point estimates.
2. **Harder states**: shift `--hops/--mix` away from one-move states; consider on-policy rounds ([ADR-0013](decisions/0013-mixed-training-state-distribution.md)). *Update 2026-11-10: a stage-3 run on hops 2-4 with a 0.6/0.2/0.2 mix is already in progress on Runpod; it needs a 50-per-hop evaluation next to stage 2 before anything can be concluded* ([ADR-0023](decisions/0023-stage-3-harder-training-states-on-runpod.md)).
3. **One cheap ablation**: flat vs ranked invalid reward (`--flat-invalid`), since it is already a flag.
4. On a new box, do a **5-step dry run** and read peak VRAM, step time, power draw, `frac_reward_zero_std` and cutoff rate before any long run.
5. Commit the launcher/chain scripts under `scripts/`; test a fresh clone after any `.gitignore` or dependency change.
6. ~~Make the Linux case work in `detect_machine()` and tag runs with the provider.~~ Done: the Runpod stage-3 run reports `machine:cuda` and `provider:runpod` ([ADR-0021](decisions/0021-try-runpod-on-an-experiment-branch.md)).
7. Untracking CLAUDE.md means new clones and pods do not inherit the standing rules; decide which of them belong in a tracked file ([ADR-0022](decisions/0022-stop-tracking-claude-md.md)).

## Open questions

* Does shaping help beyond a trajectory-only reward on this graph?
* Is the trained R1 actually better than Qwen at equal decoding and n?
* Can any recipe solve hop-4 pairs on Wikispeedia, or is the graph the limit?
