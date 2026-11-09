---
status: "proposed (run in progress; outcome unknown at time of writing)"
date: 2026-11-09
decision-makers: "project owner (user), dev agent (Claude). The discussion behind this run is not in the transcript this record was built from; the decision is reconstructed from the run's recorded configuration."
consulted: "-"
informed: "-"
---

# Stage 3: continue training from the stage-2 adapter on harder states, on Runpod

## Context and Problem Statement

After stage 2 the trained R1 reached 35.0% on the held-out pairs but solved no hop-4 games, and per-step training success (47%) overstated game success because 57% of training states were one move from the target ([ADR-0013](0013-mixed-training-state-distribution.md), [ADR-0020](0020-small-eval-on-the-gpu-box.md)). The lessons from those results named "harder states" as the first training change to try. A stage-3 run was started on the new provider ([ADR-0021](0021-try-runpod-on-an-experiment-branch.md)).

## Decision Drivers

* Train on states where the model currently fails (hops 2-4), not on one-move states.
* Reuse the stage-2 adapter instead of restarting.
* Keep everything else the same as stage 2 so the effect of the state mix is attributable.

## Considered Options

* Keep the stage-2 state mix (hops 1-3, 0.4/0.3/0.3) for more steps.
* Shift the mix toward harder states, resuming the stage-2 adapter (what the run did).
* On-policy state rounds (states from the model's own rollouts): still not implemented.

## Decision Outcome

Chosen option: "harder states, resume stage 2". Recorded configuration of W&B run `mg1m4f44` (`grpo-step-r1-1.5b-stage3-cuda-9189b84+dirty`, tags `machine:cuda`, `provider:runpod`, `stage:3`):

| setting | stage 2 | stage 3 |
|---|---|---|
| resume adapter | stage 1b final | `data/ckpt/grpo-step-r1-stage2` |
| hops | 1, 2, 3 | **2, 3, 4** |
| mix (start / optimal / noisy) | 0.4 / 0.3 / 0.3 | **0.6 / 0.2 / 0.2** |
| completion cap | 512 | 512 |
| steps, G, prompts per step | 300, 8, 4 | 300, 8, 4 |
| seed | 1 | 3 (different from earlier stages, so states are fresh) |
| lr, beta, temperature | 1e-5, 0, 0.7 | same |
| feedback, ranked invalid tiers | off, ranked | off, ranked |
| hardware | RTX 5070 Ti (Windows) | NVIDIA RTX PRO 4000 Blackwell (Runpod) |

At the last W&B heartbeat read (2026-11-09 20:02 UTC) the run was at global step 272 of 300.

### Consequences

* Good, because it directly tests whether the hop-4 and training/eval-mismatch problems are about the state distribution.
* Good, because only the state mix and hardware differ, so a gain or loss can be attributed (modulo the new GPU).
* Bad, because the training-state difficulty and the hardware changed in the same run, and no eval has been run yet; the effect is unknown until the stage-3 adapter is evaluated at 50 pairs per hop alongside stage 2.
* Bad, because the run's git sha is `9189b84+dirty`, i.e. launched from a tree with uncommitted changes (the provider tagging was not yet committed), so the exact code is not pinned to a commit.
* Open: update this ADR to accepted or rejected after the evaluation; if stage 3 does not help, the next options are on-policy rounds and the ablations listed in [LESSONS.md](../LESSONS.md).

## Implementation Plan

* **Affected paths**: `scripts/grpo_train.py` (`--stage 3 --hops 2 3 4 --mix 0.6 0.2 0.2 --seed 3 --resume-adapter data/ckpt/grpo-step-r1-stage2 --max-completion 512 --max-steps 300 --save-steps 25`), `src/wikigame/states.py`, output `data/ckpt/grpo-step-r1-stage3` on the pod.
* **Next steps**: ship the stage-3 adapter back with the artifact mechanism ([ADR-0018](0018-wandb-artifact-handoff.md)), then evaluate stage 2 and stage 3 at 50 pairs per hop with the same pairs and decoding ([ADR-0005](0005-heldout-evaluation-protocol.md)).

### Verification

- [ ] W&B run `mg1m4f44` is `finished` with 300 steps and `data/ckpt/grpo-step-r1-stage3` contains the final adapter.
- [ ] An eval of the stage-3 adapter at 50 pairs per hop is on the Baselines page next to stage 2 final.

## More Information

Provenance: W&B run `mg1m4f44` configuration and summary, read on 2026-11-10. The reason for the choice is inferred from the lessons recorded in [LESSONS.md](../LESSONS.md), not quoted from a discussion.
