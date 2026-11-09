---
status: "accepted (with a known statistical limit)"
date: 2026-11-08
decision-makers: "CUDA-box agent (Claude), accepted by the project owner via the handoff; the owner deferred the 50-per-hop R1 baseline"
consulted: "-"
informed: "dev agent (Claude)"
---

# Evaluate the four R1 checkpoints at 20 pairs per hop (80 games each)

## Context and Problem Statement

After training, four models needed evaluation on the held-out pairs: untrained R1, stage 1b final, stage 2 step 100 and stage 2 final. One eval took 0.7-1.7 h on the RTX with R1 at up to 1024 new tokens per answer, and the box was rented time.

## Decision Drivers

* Finish all four in one unattended chain ([ADR-0017](0017-unattended-chained-stages.md)).
* Keep the same pairs and metrics as the shared protocol ([ADR-0005](0005-heldout-evaluation-protocol.md)).

## Considered Options

* 50 pairs per hop (200 games per model; the same size as the Qwen baselines) - several more hours on the rented box.
* 20 pairs per hop: the first 20 of the shared 50-per-hop pool (80 games per model) (chosen).

## Decision Outcome

Chosen option: "20 pairs per hop", with `--lenient --temperature 0.6 --max-new-tokens 1024`. The user later explicitly deferred a 50-per-hop R1 baseline ("we will defer it for later").

| model | overall | h1 | h2 | h3 | h4 | invalid |
|---|---|---|---|---|---|---|
| untrained R1 | 13.8% (11/80) | 10/20 | 1/20 | 0/20 | 0/20 | 73% |
| stage 1b final (cap 1024) | 26.2% (21/80) | 16/20 | 5/20 | 0/20 | 0/20 | 52% |
| stage 2 step 100 | 33.8% (27/80) | 17/20 | 9/20 | 1/20 | 0/20 | 38% |
| stage 2 final (cap 512) | 35.0% (28/80) | 16/20 | 8/20 | 4/20 | 0/20 | 37% |

(Numbers recomputed on the Mac from the raw episode files on 2026-11-09 and matching the CUDA agent's table; stage 1b shows 26.2% by episode count versus 26.3% in its notes.)

### Consequences

* Good, because training roughly 2.5x'd success (13.8% to 35.0%) and removed most unparseable output.
* Bad, because one game is 1.25 points overall and the intervals are wide: step 100 (33.8%) versus final (35.0%) is indistinguishable, and the Mac's Qwen2.5-1.5B baseline (38.0%, n=50 per hop, greedy, different settings) is statistically comparable to the trained R1.
* Bad, because no model solves any hop-4 game.
* Follow-up: 50-per-hop R1 baseline and evals (deferred), then harder training states ([ADR-0013](0013-mixed-training-state-distribution.md)).

## Implementation Plan

* **Affected paths**: `scripts/zero_shot.py` (`--n-per-hop 20 --adapter ... --model ...`), `data/run_evals_after_stage2.ps1`, `data/runs/*.jsonl`, W&B runs `a7ldzv08`, `zw66dml9`, `en686sia`, `q98p2apa`.
* **Command**: `HF_HOME=data/hf uv run python scripts/zero_shot.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --adapter data/ckpt/<dir> --name <exp> --lenient --temperature 0.6 --max-new-tokens 1024 --n-per-hop 50` for the bigger run.

### Verification

- [ ] Each of the four JSONLs under `data/runs/` has 80 episodes.
- [ ] The Baselines page lists the four runs (job type `eval` / `baseline`).

## More Information

Provenance: CLAUDE.md "Final results" and the 2026-11-08 user message deferring the 50-per-hop benchmark. Recorded retrospectively on 2026-11-10.
