---
status: "accepted"
date: 2026-11-06
decision-makers: "project owner (user), dev agent (Claude)"
consulted: "arXiv 2610.13948 (game rules)"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Environment rules, observation and action format follow the paper

## Context and Problem Statement

We needed a gym-style environment whose rules are close to the benchmark paper (so results are interpretable) and whose output format a ~1.5B model can reliably produce and a parser can score.

## Decision Drivers

* Comparable to the paper's setup.
* Parseable, unambiguous actions; invalid output must be detectable and classifiable ([ADR-0007](0007-invalid-answer-causes-and-optional-feedback.md)).
* The observation must fit a small model's context: prompts are short (R1 tokenizer: min 185, median 307, max 792 tokens).

## Considered Options

* Free-form text answers, parsed by fuzzy matching.
* Strict tag format `<think>...</think><answer>{"title": "..."}</answer>`, only the JSON in `<answer>` is parsed.
* Function/tool-calling API.

## Decision Outcome

Chosen option: "strict tag format", with these rules in `WikiGameEnv`:

* The model sees the target's summary, the current page, steps left and the titles of the current page's links (at most **200** shown).
* No revisits; a dead end ends the game; step cap **10**; **3** invalid answers end the game.
* Unshaped eval reward: success +1.0, step cost 0.02, invalid -0.1, dead end -0.2; `extra_steps` over the shortest path is the paper's efficiency metric.
* A *lenient* parser (accepts a bare title or a fenced json block) exists only as an eval option (`--lenient`), never in training.

### Consequences

* Good, because every action has exactly one parse and one verdict.
* Bad, because a small model fails the format often: Qwen2.5-1.5B had ~60% invalid actions at zero shot, and untrained R1 94-97% ([ADR-0009](0009-switch-to-r1-distill-qwen-1-5b.md)). Format failure dominates early learning ([ADR-0011](0011-ranked-invalid-rewards-no-warm-start.md)).
* Neutral, because lenient parsing raises zero-shot success only a few points (33.5% to 38.0% for Qwen); both numbers are reported.

## Implementation Plan

* **Affected paths**: `src/wikigame/env.py` (`WikiGameEnv`, `parse_action`, `RewardConfig`), `src/wikigame/rollout.py`, `tests/test_env.py`.
* **Patterns to follow**: `SYSTEM_PROMPT` and the observation text are shared by baselines and training; changing them invalidates comparisons with old runs (the feedback line is therefore opt-in, ADR-0007).
* **Patterns to avoid**: do not widen training parsing to the lenient parser.

### Verification

- [ ] `uv run pytest tests/test_env.py` passes (`test_parse_action`, `test_step_limit_truncates`, `test_dead_end_terminates`, `test_revisit_and_unknown_are_invalid`).
- [ ] `test_flag_off_observation_unchanged` passes: default observation is byte-identical to the original.

## More Information

Provenance: built on 2026-11-06 (assistant report 08:13) and recorded retrospectively on 2026-11-10. The exact option list considered at the time was not preserved; the alternatives above are reconstructed from the design discussion.
