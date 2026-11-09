---
status: "accepted"
date: 2026-11-07
decision-makers: "project owner (user)"
consulted: "dev agent (Claude), recommended option chosen"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Sample training states from a mix of fresh starts, optimal-path prefixes and noisy walks

## Context and Problem Statement

With per-step reward ([ADR-0010](0010-per-step-progress-shaped-reward.md)) one prompt is one state, so the mix of states decides what the model learns. Zero-shot behaviour was weak: ~60% of actions invalid and rarely past hop 2.

## Decision Drivers

* Cover both the beginning of games and mid-game states.
* Include "the model's own mistakes" (states after wrong turns) without having to run the model for data collection.
* Never train on held-out evaluation pairs ([ADR-0005](0005-heldout-evaluation-protocol.md)).

## Considered Options

* Fresh start states only.
* Prefixes of shortest paths only.
* Mixed: start states + optimal-path states + policy rollouts (chosen, "Recommended").

## Decision Outcome

Chosen option: "mixed". Implemented in `states.sample_states` with `StateMix(start=0.4, optimal=0.3, noisy=0.3, hops=(1,2,3), epsilon=0.35)`. The "policy rollouts" were approximated by **noisy epsilon-greedy walks** (a stand-in for the model's own mistakes); real on-policy state rounds are an open item. `p_post_invalid` (share of states right after an invalid answer) defaults to 0.

### Consequences

* Good, because training is cheap to set up and never needs the model for data generation.
* Bad, because the sampled set skews easy: **57% of states are one move from the target**, 26% two, 17% three. This is the likely reason per-step training success (47%) is far above full-game eval success (35%), and why no model solves any hop-4 game.
* Follow-up: next run should shift `--hops/--mix` toward harder states and consider on-policy rounds.
* Gotcha: stage 1 consumed ~1200 of 1448 sampled states, so a second stage with the same seed would re-sample the same states; use a different `--seed` per stage ([ADR-0017](0017-unattended-chained-stages.md)).

## Implementation Plan

* **Affected paths**: `src/wikigame/states.py` (`StateMix`, `_noisy_prefix`, `sample_states`), `scripts/grpo_train.py` (`--hops`, `--mix`, `--p-post-invalid`, `--seed`), `tests/test_grpo_parts.py`.
* **Patterns to avoid**: do not sample any pair in `eval.heldout_pairs`.

### Verification

- [ ] `uv run pytest tests/test_grpo_parts.py` passes.
- [ ] The training script prints the kind counts at launch (e.g. `1448 training states: {'start': 793, 'optimal': 282, 'noisy': 373}`).

## More Information

Provenance: AskUserQuestion answer 2026-11-07 20:31, commit 7230640. Recorded retrospectively on 2026-11-10.
