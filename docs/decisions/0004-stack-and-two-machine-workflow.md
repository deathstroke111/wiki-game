---
status: "accepted"
date: 2026-11-06
decision-makers: "project owner (user), dev agent (Claude)"
consulted: "-"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Plain PyTorch + TRL in a `uv` venv; Mac for dev, rented GPU for training, with a machine-detection hook

## Context and Problem Statement

Two machines are in play: a MacBook M4 (24 GB unified memory, MPS) for development and a rented NVIDIA box (RTX, 16 GB VRAM, 64 GB RAM) for training. The rented box is "not always available". The project owner's standing instruction (CLAUDE.md): install and run everything inside the project Python environment, never globally, and always check which machine a session is on using a hook.

## Decision Drivers

* Real training needs CUDA (and 16 GB is the budget); the Mac cannot train a 1.5B model usefully (R1 generation alone was ~25 s per call on MPS).
* Code must run unchanged on both machines.
* Agents on each machine must know where they are before doing anything heavy.

## Considered Options

* MLX on the Mac (considered first, because the Mac is the dev box).
* Plain PyTorch + TRL, device chosen at runtime (MPS on the Mac, CUDA on the box).

## Decision Outcome

Chosen option: "plain PyTorch + TRL", because training happens on CUDA; MLX was dropped on 2026-11-06. Concretely:

* `uv` manages one project `.venv`; always `uv run ...`; ad-hoc scripts that read data use `uv run python -I`.
* `.claude/hooks/detect-machine.sh` (registered as a `SessionStart` hook in `.claude/settings.json`) prints `LOCAL_DEV` / `CUDA_RENTED` / `UNKNOWN`; `wikigame.tracking.detect_machine()` mirrors it (override: `WIKIGAME_MACHINE=local|cuda`). Runs are tagged `machine:*`.
* Torch comes from a CUDA 13.2 index on Windows/Linux (`pytorch-cu132`, needed for Blackwell sm_120) and from PyPI on macOS (MPS).

### Consequences

* Good, because one code path serves both machines and W&B runs say which machine produced them.
* Bad, because the CUDA branch of the hook was written on the Mac and had never run on a real box; the rented box turned out to be **Windows**, not Linux, so the hook and `detect_machine()` needed a fix (commit d6a73b3) and uv initially resolved a CPU-only torch wheel there (commit 27d9e7c).
* Bad, because `uv sync --group A --group B` uninstalls every group not named. On the Mac, syncing `llm`+`dev` removed streamlit (group `app`) and broke the app tests until `--group app` was added (2026-11-09).
* Follow-up: Linux + working `nvidia-smi` must also count as CUDA for Runpod ([ADR-0021](0021-try-runpod-on-an-experiment-branch.md)).

## Implementation Plan

* **Affected paths**: `.claude/hooks/detect-machine.sh`, `.claude/settings.json`, `src/wikigame/tracking.py` (`detect_machine`), `pyproject.toml` (groups `llm`, `dev`, `app`; torch index pin), `uv.lock`.
* **Commands**: Mac: `uv sync --group llm --group dev --group app`. CUDA box: `uv sync --group llm --group dev`.
* **Patterns to avoid**: global `pip install`; `uv sync` without every group you still need.

### Verification

- [ ] The session-start output names the machine correctly on each machine.
- [ ] `uv run python -c "from wikigame.tracking import detect_machine; print(detect_machine())"` prints `local` on the Mac and `cuda` on the GPU box.
- [ ] `uv run pytest -q` passes on the Mac with all three groups (78 tests at last run).

## More Information

Provenance: assistant report of 2026-11-06 08:13 ("Plan change"), CLAUDE.md "Windows / RTX lessons". Recorded retrospectively on 2026-11-10.
