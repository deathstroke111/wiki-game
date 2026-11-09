---
status: "accepted"
date: 2026-11-06
decision-makers: "project owner (user)"
consulted: "dev agent (Claude) laid out the trade-offs"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Per-step reward with progress shaping (potential = minus BFS distance to the target)

## Context and Problem Statement

How should the policy be rewarded? The user asked for the trade-offs of three options: whole-trajectory, short sub-sequences, or every next step.

## Decision Drivers

* Sparsity: a random policy gets 0% at 2+ hops, so with trajectory-level reward most GRPO groups have zero reward variance and no gradient.
* Credit assignment: one bad move should not taint a winning trajectory.
* We have exact BFS distances, so progress can be computed exactly.
* Eval metrics must stay honest (unshaped).

## Considered Options

* Whole trajectory (true objective; simplest; blurry credit; very sparse for a small model).
* Short segments of k moves (medium on every axis; highest complexity: branching rollouts).
* Every step (exact credit, no sparsity, moderate complexity; uses privileged distance information).

## Decision Outcome

Chosen option: "every step", main run, on 2026-11-06 ("Okay lets start with per step runs then"). A move scores:

* +1 if it gets closer to the target, 0 sideways, -1 farther (clipped to +-1; unknown distance counts as -1), times a shaping weight of 1.0;
* minus step cost 0.02; success +1; dead end -0.2;
* invalid answers: see [ADR-0011](0011-ranked-invalid-rewards-no-warm-start.md).

Shaping is a `RewardConfig` option (`shaping=0.0` by default, so old numbers reproduce exactly); training uses `shaping=1.0`. Eval reward stays unshaped ([ADR-0005](0005-heldout-evaluation-protocol.md)). A trajectory-only ablation (separate W&B group) and segment rewards are deferred.

### Consequences

* Good, because every group has signal from step 1 and potential-based shaping cannot be gamed by looping.
* Bad, because it distils the BFS oracle (privileged information) rather than learning pure exploration, and is myopic about dead ends and the no-revisit rule.
* Bad, because per-step success overstates game success: training per-step success reached 47% while full-game eval success is 35%, since 57% of training states are one move from the target ([ADR-0013](0013-mixed-training-state-distribution.md)).
* Follow-up: the trajectory-only ablation was never run, so how much shaping buys is unmeasured.

## Implementation Plan

* **Affected paths**: `src/wikigame/env.py` (`RewardConfig.shaping`, `_dist_for` cache, `info["shaping_reward"]`), `src/wikigame/grpo_reward.py` (`default_reward_config`), `src/wikigame/webapp/analysis.py` (`reward_breakdown` adds "progress shaping" only when the run config has shaping), `tests/test_env.py`.
* **Patterns to follow**: new reward terms go in `RewardConfig` with a default that leaves old rewards unchanged.

### Verification

- [ ] `uv run pytest tests/test_env.py` passes (`test_shaping_off_by_default_reproduces_old_rewards`, `test_shaped_reward_ordering_closer_sideways_farther_invalid`, `test_shaping_is_clipped_and_recorded`).
- [ ] The Diagnostics page reward components reproduce each episode's logged total (residual 0).

## More Information

Provenance: assistant trade-off table of 2026-11-06 11:11, user reply 11:33, commit f8f3528. Recorded retrospectively on 2026-11-10.
