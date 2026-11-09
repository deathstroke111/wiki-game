---
status: "accepted"
date: 2026-11-07
decision-makers: "project owner (user)"
consulted: "dev agent (Claude)"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Run the zero-shot baselines at 50 pairs per hop, on the Mac, after the environment change

## Context and Problem Statement

The first zero-shot numbers (Qwen2.5-1.5B-Instruct, commit f65c0dd) used only 10 pairs per hop, so the confidence interval was about +-30 points and it was impossible to say whether lenient parsing or any change helped. The invalid-action work ([ADR-0007](0007-invalid-answer-causes-and-optional-feedback.md)) changed the environment, so old runs needed re-running anyway.

## Decision Drivers

* Statistical resolution good enough to compare four variants (strict, lenient, each with and without feedback).
* Baselines must use the final environment so they stay comparable.
* The Qwen model costs ~1 s per call on the Mac, so a larger run is cheap there.

## Considered Options

* Keep 10 pairs per hop (fast, too noisy).
* 50 pairs per hop on the Mac after the env change (chosen; ~10-15 min per run).
* Run the large baseline on the CUDA box (not available at the time).

## Decision Outcome

Chosen option: "50 pairs per hop on the Mac after the change", four sequential runs on the same first-50 held-out pairs as random/oracle.

| policy | h1 | h2 | h3 | h4 | overall |
|---|---|---|---|---|---|
| random | 6% | 0 | 0 | 0 | 1.5% |
| zero-shot strict | 88% | 36% | 10% | 0% | 33.5% |
| zero-shot lenient | 98% | 44% | 6% | 4% | 38.0% |
| strict + feedback | 94% | 44% | 12% | 4% | 38.5% |
| lenient + feedback | 98% | 56% | 10% | 4% | 42.0% |
| oracle | 100% | 100% | 100% | 100% | 100% |

### Consequences

* Good, because it showed ~60% of Qwen's actions are invalid, which drove ADR-0007, [ADR-0009](0009-switch-to-r1-distill-qwen-1-5b.md) and [ADR-0011](0011-ranked-invalid-rewards-no-warm-start.md).
* Bad, because the 10-per-hop numbers were too optimistic; they are kept (not deleted) in `data/runs/superseded-10-per-hop/` and are superseded in the app by "latest per policy".
* Bad, because no *R1* baseline was run before training (the owner chose "see how it performs after the training"); an untrained R1 baseline was only produced later on the CUDA box at 20 pairs per hop ([ADR-0020](0020-small-eval-on-the-gpu-box.md)). A 50-per-hop R1 baseline is still deferred.

## Implementation Plan

* **Affected paths**: `scripts/zero_shot.py` (`--n-per-hop 50`, `--lenient`, `--feedback`), `data/runs/*.jsonl` (gitignored), W&B runs `8kbhatpb`, `oktwkf89`, `4y6hjeaz`, `n1q4dqv4`.
* **Patterns to follow**: run sequentially in the background; pass flags as separate arguments (a zsh loop once passed `"--lenient --feedback"` as one argument because zsh does not word-split unquoted variables).

### Verification

- [ ] The Baselines page shows these four rows with no `n_per_hop` mismatch warning.
- [ ] `wc -l` of each 50-per-hop JSONL is 200 episodes.

## More Information

Provenance: AskUserQuestion "Run on the Mac after the env change"; CLAUDE.md baselines table. Recorded retrospectively on 2026-11-10.
