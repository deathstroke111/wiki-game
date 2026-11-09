---
status: "accepted"
date: 2026-11-07
decision-makers: "project owner (user)"
consulted: "dev agent (Claude) supplied the zero-shot evidence"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Classify why answers are invalid; make error feedback an opt-in environment flag

## Context and Problem Statement

The first zero-shot baseline (Qwen2.5-1.5B-Instruct, 10 pairs per hop) failed almost only through invalid actions: every failed game ended with three invalid answers. The model often named the target page even when it was not linked from the current page (39% of actions), re-picked the current page (13%, later corrected to 8-12% on the 50-per-hop runs), and repeated the same wrong answer (18 of 40 games) because the environment said nothing after an invalid action. The paper's setup retries with a message.

## Decision Drivers

* Understand *why* actions fail, not only that they fail.
* Keep old baselines comparable (default observation must not change).
* Measure whether feedback actually helps before baking it into training.

## Considered Options

* Add the feedback line to the observation unconditionally.
* Add it as an environment flag, default off, and run baselines both ways (chosen).
* No feedback (status quo).

## Decision Outcome

Chosen option: "env flag, default off, run both":

* `classify_invalid(...)` returns one of `unparseable`, `unknown_title`, `current_page`, `revisit`, `target_not_linked` (target named but not linked), `not_linked`, or `None` for a legal move; `info["cause"]` and each recorded step carry it.
* `WikiGameEnv(feedback=True)` inserts "Your last answer was invalid: ..." (cause-specific, with remaining attempts) in the observation directly after an invalid action only.
* The app shows causes: a Diagnostics cause chart and a per-step "why invalid" column in Episodes (the user chose "Diagnostics plus per-step tag"). Old episodes without a stored cause are backfilled with the same classifier.
* The training script exposes `--feedback` and `--p-post-invalid`; the final R1 runs used neither.

### Consequences

* Good, because the dominant failure became measurable: target-naming is 40-47% of all actions for Qwen.
* Evidence (Qwen2.5-1.5B, 50 pairs per hop, greedy): feedback lifts success only slightly (strict 33.5% to 38.5%, lenient 38.0% to 42.0%) and the confidence intervals overlap; repeated-identical-invalid games fell from 141 to 118 (strict) and 120 to 106 (lenient) of 200. The model mostly ignores the message.
* Bad, because feedback states exist only after an invalid answer, so training with feedback would need post-invalid states in the dataset (not done).

## Implementation Plan

* **Affected paths**: `src/wikigame/env.py` (`classify_invalid`, `_FEEDBACK`, `feedback`), `src/wikigame/rollout.py`, `scripts/zero_shot.py` (`--feedback`), `src/wikigame/webapp/analysis.py` (`CAUSE_INFO`, `step_causes`, `invalid_cause_table`), `app/views/diagnostics.py`, `app/views/episodes.py`.
* **Patterns to follow**: message text is built with `.replace("{a}", ...)`, not `str.format` (braces in the unparseable message once raised a `KeyError`).

### Verification

- [ ] `uv run pytest tests/test_env.py` passes (`test_invalid_causes`, `test_feedback_only_after_invalid_and_only_when_enabled`).
- [ ] `uv run pytest tests/test_app_analysis.py` covers cause backfill on an old-format episode.

## More Information

Provenance: AskUserQuestion answers of 2026-11-07 ("Feedback as an env flag, run both"; "Diagnostics plus per-step tag in the explorer"); commits 36c3926 and 556db92. Recorded retrospectively on 2026-11-10.
