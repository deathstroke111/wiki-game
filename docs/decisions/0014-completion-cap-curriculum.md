---
status: "accepted"
date: 2026-11-07
decision-makers: "project owner (user)"
consulted: "dev agent (Claude)"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Two-stage completion-length curriculum: 1024 tokens, then 512

## Context and Problem Statement

R1 averages ~700 tokens per answer and about half were cut off at 1024 even with a brevity hint; an answer cut off before `</answer>` counts as invalid. The memory and time cost on the 16 GB GPU scales with group size x completion length. The user asked: "is it possible to initially keep the CAP 1024, G=8 and then later shift to Cap=512, G=8? or will there be too much messy code/data?"

## Decision Drivers

* Give the model room to learn the format at first, then force concision (and cheaper steps).
* Minimal extra code.

## Considered Options

* A fixed cap of 1024 throughout.
* A fixed cap of 512 from the start (cheaper, but most early answers would be cut off).
* Stage 1 at 1024, stage 2 at 512 resuming the adapter (chosen, the user's idea).

## Decision Outcome

Chosen option: "two stages". `--stage` labels the run (`grpo-step-r1-stage<N>` in W&B group `grpo-per-step`), `--max-completion` sets the cap, `--resume-adapter` continues from the previous adapter (weights only, no optimizer state). Eval uses a fixed `--max-new-tokens 1024` in both cases.

What actually ran: stage 1 was killed at step 32 (too slow, see [ADR-0016](0016-windows-rtx-run-rules.md)); "stage 1b" resumed from its `checkpoint-25` for 275 steps (effective stage 1 = 25 + 275 = 300), then stage 2 (cap 512, 300 steps, seed 1).

### Consequences

* Good, because stage 2 is faster (~40-55 s per step versus ~85 s) and cutoff fell from 53% to 12% (stage 1b) to 1% (stage 2); mean completion length 838 to ~520 to ~179 tokens.
* Good, because the result held up: eval success 26.3% (stage 1b) to 33.8% (stage 2 step 100) to 35.0% (stage 2 final), n=20 per hop.
* Bad, because there is no control run at 512 from the start, so how much the curriculum helped is unmeasured; the CIs are also wide ([ADR-0020](0020-small-eval-on-the-gpu-box.md)).
* Neutral, because optimizer state is lost at each resume.

## Implementation Plan

* **Affected paths**: `scripts/grpo_train.py` (`--stage`, `--max-completion`, `--resume-adapter`, `--out`), `src/wikigame/grpo_reward.py` (cutoff detection uses the cap).
* **Commands**: `uv run python scripts/grpo_train.py --stage 1 --max-completion 1024 --max-steps 300` then `--stage 2 --max-completion 512 --resume-adapter data/ckpt/<stage1-dir> --seed 1`.

### Verification

- [ ] Final adapters exist in `data/ckpt/grpo-step-r1-stage1b/` and `data/ckpt/grpo-step-r1-stage2/` (`uv run python scripts/publish_handoff.py --phase training --verify .` prints `all 56 files verified OK`).
- [ ] W&B runs `kkqk572s` (1b) and `sa7pbhpc` (2) show `completions/clipped_ratio` falling.

## More Information

Provenance: user question in the 2026-11-07 20:31 dialog; run history in CLAUDE.md. Recorded retrospectively on 2026-11-10.
