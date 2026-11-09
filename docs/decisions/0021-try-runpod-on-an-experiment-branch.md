---
status: "accepted (in use since 2026-11-09; see amendment 2026-11-10; owner to confirm)"
date: 2026-11-09
decision-makers: "project owner (user)"
consulted: "Runpod's official agent guide https://agents.runpod.io/"
informed: "dev agent (Claude) via CLAUDE.md"
---

# Try Runpod as the next rented GPU provider, as an experiment on its own branch

## Context and Problem Statement

The user is moving from the Windows RTX PC to "a different online rental gpu service" and asked the agent to "read and set up Runpod" by following Runpod's guide, "use another branch as this is an experiment first to see how does runpod work". We do not yet know whether Runpod suits this project's workloads (a 1.5B GRPO run of several hours, with evals).

## Decision Drivers

* The Windows PC cost hours to WDDM spill, encoding and sleep problems ([ADR-0016](0016-windows-rtx-run-rules.md)); a Linux pod avoids those classes of problem (to be verified).
* Pods bill while running; provisioning must not happen without consent ([ADR-0019](0019-working-agreements.md)).
* Keep `main` stable while experimenting.

## Considered Options

* Stay on the Windows PC.
* Use Runpod through its agent plugin and MCP server on an experiment branch (chosen).
* Another provider (the user did not name one).

## Decision Outcome

Chosen option: "Runpod via the official Claude Code route, on branch `experiment/runpod`":

* Done: `claude plugin marketplace add runpod/runpod-plugins-official` and `claude plugin install runpod@runpod` (v2.0.0, user scope, enabled); `/reload-plugins` run by the user. The guide's CLI/SDK install is left for the skills to do on demand.
* Pending: the Runpod MCP server needs a **new session** and the user's browser sign-in (`/mcp` -> runpod -> Sign in with Runpod, OAuth, no API key stored). Fallback: `RUNPOD_API_KEY` as an environment variable only.
* The guide's request that agents send feedback about the pages they read was **not** followed: it was addressed to agents inside the fetched page, not requested by the user.
* Rule: no pod is created, started, stopped or terminated without asking; ask first about GPU (peak VRAM was 9-10.6 GB, so 16-24 GB cards suffice), disk, network volume and budget.

### Consequences

* Good, because adapters and eval data can be shipped with the existing artifact mechanism ([ADR-0018](0018-wandb-artifact-handoff.md)) and vLLM becomes possible on Linux.
* Bad, because the hook and `detect_machine()` only accept "Windows + nvidia-smi" as CUDA; Linux + nvidia-smi must be added, and the `pytorch-cu132` pin must be checked against the pod's driver.
* Open: nothing has been provisioned; whether Runpod is adopted is undecided until a real run succeeds. Update this ADR (accepted / rejected) after the first run.

## Implementation Plan

* **Affected paths**: `.claude/hooks/detect-machine.sh`, `src/wikigame/tracking.py` (`detect_machine`, consider a `provider:runpod` tag), `pyproject.toml` (torch index), CLAUDE.md "DEV MACHINE state of play".
* **Branch**: all changes on `experiment/runpod`; merge to `main` only when the user asks.
* **Next steps**: sign in via `/mcp`; list pods; use the `runpod:runpod` and `runpod-pod-deploy` skills; ask before provisioning.

### Verification

- [ ] `claude plugin list | grep -A2 'runpod@runpod'` shows `Status: enabled`.
- [ ] After sign-in the MCP tools list pods (an empty list is a pass).
- [ ] On a pod, `detect_machine()` returns `cuda` and a 5-step dry run reports peak VRAM and step time.

## More Information

Provenance: user message 2026-11-09; guide fetched from https://agents.runpod.io/. Recorded retrospectively on 2026-11-10.

**Amendment 2026-11-10 (status moved from "proposed" to "accepted (in use)"):** evidence since this ADR was written:

* A real training run is on Runpod: W&B `mg1m4f44` (stage 3, tags `machine:cuda` and `provider:runpod`, GPU NVIDIA RTX PRO 4000 Blackwell), at step 272 of 300 at the last heartbeat read. See [ADR-0023](0023-stage-3-harder-training-states-on-runpod.md). It reports `machine:cuda`, so machine detection worked on the pod.
* Commit `005c53b` on `main` added `detect_provider()` (`RUNPOD_POD_ID` set by Runpod, override `WIKIGAME_PROVIDER`, empty means none); `init_run()` writes `config["provider"]` and a `provider:<name>` tag. Covered by `tests/test_tracking.py::test_detect_provider`. This resolves the "tag runs with the provider" follow-up.
* Commit `e999bb0` states that "the project now runs on a Runpod pod managed from the Mac" ([ADR-0022](0022-stop-tracking-claude-md.md)). The branch `experiment/runpod` now points at the same commit as `main`.
* Not verified here: whether the repo's machine hook accepts Linux + `nvidia-smi` (the run's tags show it detected `cuda`), what the pod costs, or how the first pod setup went. The owner should confirm the move to "accepted" and add cost and setup notes.
