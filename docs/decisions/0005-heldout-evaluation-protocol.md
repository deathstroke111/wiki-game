---
status: "accepted"
date: 2026-11-06
decision-makers: "project owner (user), dev agent (Claude)"
consulted: "-"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# One shared held-out evaluation protocol for every policy

## Context and Problem Statement

Random, oracle, prompted-LLM, human and trained policies all need to be compared on the same footing, and training must never see the evaluation pairs.

## Decision Drivers

* Honest comparisons across baselines and checkpoints.
* Exact-hop difficulty control (1-4 hops).
* Small samples are noisy: 10 games per hop gave confidence intervals of about +-30 points ([ADR-0008](0008-baselines-at-50-pairs-per-hop.md)).

## Considered Options

* Evaluate on whatever pairs each script samples.
* A fixed, seeded pool of held-out (source, target) pairs shared by all scripts, excluded from training.

## Decision Outcome

Chosen option: "fixed shared pool":

* `eval.heldout_pairs`: seed **1234**, `EVAL_POOL` = **50** pairs per hop, hops **1-4**. `scripts/zero_shot.py --n-per-hop N` takes the first N of each hop's pool, so smaller runs are subsets of larger ones.
* Metrics: `success@k` (per hop), `extra_steps@k`, `n@k`, `invalid_rate`, `invalid_cause/<cause>` (share of all actions).
* Eval reward is always **unshaped** ([ADR-0010](0010-per-step-progress-shaped-reward.md)); the Baselines page shows Wilson intervals.
* `states.sample_states` never samples a held-out pair ([ADR-0013](0013-mixed-training-state-distribution.md)).
* Baselines: random, oracle (ceiling, 100%), prompted LLM, human (Play page).

### Consequences

* Good, because any checkpoint can be dropped onto the Baselines page next to every baseline.
* Good, because the oracle at 100% and random at 1.5% bound the scale.
* Bad, because the pool is small (50 per hop); at 20 per hop one game is 1.25 points overall ([ADR-0020](0020-small-eval-on-the-gpu-box.md)).

## Implementation Plan

* **Affected paths**: `src/wikigame/eval.py` (`EVAL_POOL`, `heldout_pairs`, `summarize`), `scripts/zero_shot.py`, `scripts/smoke_run.py`, `src/wikigame/webapp/analysis.py` (Wilson CIs), `tests/test_rollout_eval.py`.
* **Patterns to avoid**: do not change the seed or pool size; do not use shaped reward for reporting.

### Verification

- [ ] `uv run pytest tests/test_rollout_eval.py tests/test_grpo_parts.py` passes (held-out exclusion is exercised in the state-sampler tests).
- [ ] The Baselines page warns when two runs have different `n_per_hop`.

## More Information

Provenance: CLAUDE.md "What exists" and the 2026-11-06/07 session. Recorded retrospectively on 2026-11-10.
