---
status: "accepted"
date: 2026-11-07
decision-makers: "project owner (user)"
consulted: "dev agent (Claude) raised the cold-start risk"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Rank invalid answers by cause in the training reward; no warm start

## Context and Problem Statement

With shaping on ([ADR-0010](0010-per-step-progress-shaped-reward.md)) the worst valid move is about -1.22 (farther -1, step cost, dead end). Invalid answers must score below that or the model could prefer a bad move to a format error. A fixed floor (-2 for everything invalid) was the first choice. But early R1 outputs were almost all invalid, so a flat floor gives most GRPO groups identical rewards (no gradient). A Qwen 0.5B smoke run confirmed it: all answers were unparseable, reward std 0, loss and grad_norm 0.

## Decision Drivers

* Cold start: untrained R1 had 94-97% invalid actions (about 90% unparseable).
* Give GRPO a gradient between "closer to a legal answer" and "garbage".
* The owner explicitly ruled out supervised warm-starting.

## Considered Options

* Fixed floor of -2 for all invalid answers (chosen first, 2026-11-07, "Fixed floor below any valid move").
* Ranked tiers by how close the answer was to a legal move (chosen).
* SFT warm start on oracle moves (rejected by the owner: "Do not warm start.").

## Decision Outcome

Chosen option: "ranked tiers", all below the worst valid move:

| invalid cause | reward |
|---|---|
| real page, not a legal move (`not_linked`, `target_not_linked`, `current_page`, `revisit`) | -1.5 |
| non-existent title (`unknown_title`) | -1.75 |
| unparseable | -2.0 |
| cut off at the token cap with no `</answer>` | -2.25 (extra -0.25 on top of unparseable) |

`--flat-invalid` restores a single -2 for an ablation (not run). The owner's reasoning: "maybe ranking answers helps shape the reward function ... award different answers differently". This ADR refines the earlier fixed-floor choice.

### Consequences

* Good, because training learned to finish and emit the tag first: stage-1b invalid rate fell 96% to 36% and cutoff 53% to 12%; stage 2 ended at invalid 20%, cutoff 1%.
* Bad, because the tier ordering is a judgement call and the flat-vs-ranked ablation was never run, so its effect is unproven.
* Bad, because tiers must stay below the worst valid move; changing shaping weights or the dead-end penalty can silently break the ordering.

## Implementation Plan

* **Affected paths**: `src/wikigame/env.py` (`RewardConfig.invalid_by_cause`, `shaped_invalid`), `src/wikigame/grpo_reward.py` (`StepReward`, `cutoff_penalty=-0.25`), `scripts/grpo_train.py` (`--flat-invalid`), `tests/test_env.py`, `tests/test_grpo_parts.py`.
* **Patterns to follow**: keep every invalid tier strictly under `-shaping - step_cost - dead_end`.

### Verification

- [ ] `uv run pytest tests/test_env.py::test_invalid_tiers_are_ranked_and_flat_ablation` passes.
- [ ] Training logs show `env/invalid/<cause>` and `env/cutoff_rate` falling over steps (W&B runs `kkqk572s`, `sa7pbhpc`).

## More Information

Provenance: AskUserQuestion answer "Fixed floor below any valid move" (2026-11-07 18:02), user message 2026-11-07 20:53, commit 6774f2f. Recorded retrospectively on 2026-11-10.
