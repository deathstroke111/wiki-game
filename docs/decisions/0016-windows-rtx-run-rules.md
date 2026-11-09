---
status: "accepted (Windows-specific; partly superseded in practice by the move to Runpod, ADR-0021)"
date: 2026-11-08
decision-makers: "project owner (user), CUDA-box agent (Claude) with the user's approval for launches"
consulted: "dev agent (Claude) via CLAUDE.md"
informed: "dev agent"
---

# Operating rules learned on the Windows RTX box

## Context and Problem Statement

The "rented CUDA box" turned out to be a Windows 11 PC (RTX 5070 Ti 16 GB, 64 GB RAM), not Linux. The first stage-1 launches lost about 3 hours to three separate platform problems; the rules below were adopted to prevent a repeat.

## Decision Drivers

* The box can vanish or sleep; long runs must survive tool timeouts and be resumable.
* Time lost to silent slowdowns is expensive.

## Considered Options

Each rule had a rejected alternative:

* VRAM spill: tune `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb` (rejected: did **not** help) versus capping the per-process memory fraction (chosen).
* Console encoding: leave default cp1252 (crashed) versus UTF-8 everywhere (chosen).
* Checkpoint retention: keep newest N versus keep all (chosen).

## Decision Outcome

Chosen rules:

1. **Never run a second GPU job while training or evaluating.** At step ~10 of the first stage-1 run a 1-step resume test made PyTorch's CUDA cache balloon to 15.3 GB dedicated + 4.2 GB spilled into shared system RAM (Windows WDDM silently pages over PCIe). Steps went from 85 s to 380-480 s with the GPU "99% utilized" but only ~87 W of 300 W, and never recovered. Diagnose with `nvidia-smi` power draw and `Get-Counter '\GPU Process Memory(*)'`.
2. `grpo_train.py --gpu-mem-fraction` (default **0.75**, `torch.cuda.set_per_process_memory_fraction`) so PyTorch frees its cache instead of spilling; steps returned to 83-93 s at ~10.6 GB.
3. Always run with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` (TRL's `log_completions` printed 'ố' and crashed the first launch with `UnicodeEncodeError`); ad-hoc scripts use `python -I -X utf8`.
4. Launch long jobs detached via PowerShell `Start-Process` with logs under `data/logs/` so no tool timeout can kill them.
5. `--save-total-limit` default **0** = keep every checkpoint (every one can be evaluated later); checkpoint every 25 steps.
6. Torch from the CUDA 13.2 index on win32/linux (Blackwell sm_120 needs CUDA >= 12.8); no vLLM on Windows.
7. The PC can sleep despite "never" settings (it slept 20:11-20:43 on 2026-11-08); check Kernel-Power events 42/107 when progress stalls.

### Consequences

* Good, because after the fix stage 1b and stage 2 ran unattended and cleanly (275/275 in 8h51m; 300/300 in 4h35m).
* Bad, because the first stage-1 run was killed at step 32 and restarted from `checkpoint-25` as "stage 1b" ([ADR-0014](0014-completion-cap-curriculum.md)); the aborted runs left W&B clutter ([ADR-0019](0019-working-agreements.md)).
* Neutral, because rules 1, 2, 3, 6 and 7 are Windows/WDDM specifics that should not apply on a Linux pod ([ADR-0021](0021-try-runpod-on-an-experiment-branch.md)); verify instead of assuming.

## Implementation Plan

* **Affected paths**: `scripts/grpo_train.py` (`--gpu-mem-fraction`, `--save-total-limit`, `Telemetry` callback logging `sys/cuda_peak_gb`), `pyproject.toml` (torch index pin), CLAUDE.md "Windows / RTX lessons".
* **Patterns to follow**: watch `sys/cuda_peak_gb` and step time in the first steps of any new GPU environment.

### Verification

- [ ] `grep -n "gpu-mem-fraction" scripts/grpo_train.py` shows default 0.75.
- [ ] W&B `sys/cuda_peak_gb` for runs `kkqk572s` and `sa7pbhpc` stays under ~10.6 GB and step time stays near 85 s / 40-55 s.

## More Information

Provenance: CLAUDE.md "CUDA box: state of play" written by the RTX-side agent. Recorded retrospectively on 2026-11-10.
