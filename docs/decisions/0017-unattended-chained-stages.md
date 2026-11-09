---
status: "accepted"
date: 2026-11-08
decision-makers: "project owner (user) approved; CUDA-box agent (Claude) implemented"
consulted: "-"
informed: "dev agent (Claude) via CLAUDE.md"
---

# Chain stage 2 and the evals to start unattended, gated on the previous stage's final adapter

## Context and Problem Statement

Stage 1b needed ~9 h and stage 2 ~4.5 h; the user was away and the box is rented time. Waiting for a human to launch each next step would waste GPU hours.

## Decision Drivers

* Do not idle the rented GPU between stages.
* Do not start stage 2 from a broken stage 1 result.
* Evals compete for the same GPU and must never overlap training ([ADR-0016](0016-windows-rtx-run-rules.md), rule 1).

## Considered Options

* Manual launch of every stage and eval.
* A watcher script that waits for the previous launcher PID, checks that `wikigame_config.json` and `adapter_model.safetensors` exist, then launches the next stage (chosen).
* Run evals during training (rejected: GPU contention and the VRAM-spill slowdown).

## Decision Outcome

Chosen option: "gated chain". The user approved auto-starting stage 2 and ruled "evals only AFTER training". Watchers: `data/run_stage2_after_stage1b.ps1` and `data/run_evals_after_stage2.ps1` (gitignored). Each stage uses a different `--seed` on purpose: seed 0 would re-sample the same 1448 states that stage 1 already consumed (~1200); stage 1b used seed 2 and stage 2 seed 1.

### Consequences

* Good, because stage 2 started at 15:08 immediately after stage 1b finished cleanly, and the four evals ran back to back (0.7-1.7 h each).
* Good, because if stage 1 had died the chain would log it and not start stage 2.
* Bad, because the watcher scripts are gitignored, so they are only documented in CLAUDE.md, not reproducible from the repo.
* Bad, because an unattended chain also ran through a PC sleep (20:11-20:43): the eval froze and resumed intact, but wall-clock timings jumped.

## Implementation Plan

* **Affected paths**: `data/run_stage2_after_stage1b.ps1`, `data/run_evals_after_stage2.ps1`, `data/logs/chain.log`, `data/logs/evalchain.log`.
* **Patterns to follow**: gate on the artifact the next stage actually needs; log every state change with a timestamp.
* **Follow-up**: if chaining is reused (for example on Runpod) commit a generic launcher script under `scripts/` so it is reproducible.

### Verification

- [ ] `data/logs/chain.log` shows "stage 1b finished cleanly; starting stage 2".
- [ ] No GPU eval ran while training was active (eval W&B runs start after stage 2 ended).

## More Information

Provenance: CLAUDE.md "CUDA box notes". Recorded retrospectively on 2026-11-10.
