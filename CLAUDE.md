Always execute and install anything you need in this python environment and not globally. 
There are two environments one environment is the local dev environment(macbook m4 24 GB RAM) and other is other machine(which is rented not always available) an CUDA (NVidia RTX with 16 GB VRAM and 64 GB RAM). Always check which machine is on, using a hook - create a hook if not present. 

# Running notes (written 2026-11-08 on the MacBook, for the agent on the CUDA box)

Goal: reproduce an RL pipeline for the Wikipedia game (navigate start article -> target by clicking links) on an
**offline link graph**, inspired by arXiv 2610.13948 (benchmark, no RL) and the `aigym` project (proprietary, so all
code here is our own). Everything below is the state of play; the code is the source of truth for structure.

## Working style (the user's preferences)
- Go slowly, commit at checkpoints, consult the user before building features or making design choices.
  Present options with trade-offs; they answer via questions, then you build.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Never push unless asked.
- `uv run ...` for everything (project `.venv`). Python scripts that read data use `uv run python -I` (+ `sys.path.insert(0,'src')`
  in ad-hoc scripts). Model cache: `HF_HOME=data/hf`. W&B: team/entity **aiwizards**, project **wiki-game**.
- Machine hook (`.claude/hooks/detect-machine.sh`) prints LOCAL_DEV / CUDA_RENTED / UNKNOWN at session start.
  `wikigame.tracking.detect_machine()` mirrors it (override: `WIKIGAME_MACHINE=local|cuda`). **The CUDA branch has never
  run on a real CUDA box: confirm the hook says CUDA_RENTED and detect_machine() returns "cuda" first.**
- The rented box can vanish: checkpoint and log often. zsh does not word-split unquoted `$var` (a loop that passed
  `"--a --b"` as one argument bit us once).

## First steps on the CUDA box
1. `git pull`; `uv sync --group llm --group dev` (add `--group app` only to run the Streamlit app).
2. `data/` is gitignored. Copy `data/wikispeedia.json` (2.4 MB) from the Mac (scp), or rebuild it with
   `uv run python -I -m wikigame.data wikispeedia` after placing the SNAP Wikispeedia graph + plaintext articles in
   `data/raw/wikispeedia_graph` and `data/raw/wikispeedia_text`. Do NOT copy `data/hf` (7 GB); models download on first use.
3. `wandb login` (credentials live in ~/.netrc, never in the repo). Runs are tagged `machine:cuda`.
4. `uv run pytest -q` (74 tests pass on the Mac; none need a GPU).
5. Dry-run GRPO for a few steps and read the logs before any long run (see "GRPO" below).

## What exists (all committed on `main`)
- Env: `WikiGraph` (Wikispeedia: ~4.6k articles), `WikiGameEnv` (no revisits, dead end ends the game, step cap 10,
  3 invalid answers end the game, <=200 links shown), `set_state` rebuilds a mid-game state. Action format:
  `<think>..</think><answer>{"title": "..."}</answer>`; only the `<answer>` JSON is parsed.
- Invalid answers have a cause (`classify_invalid`): unparseable, unknown_title, current_page, revisit,
  target_not_linked (names the unlinked target), not_linked. `WikiGameEnv(feedback=True)` adds a "your last answer was
  invalid because..." line to the next observation (default off; baselines were run both ways).
- Eval: held-out pairs (seed 1234, `EVAL_POOL`=50 per hop, hops 1-4); `success@k`, `extra_steps@k`, `n@k`,
  `invalid_rate`, `invalid_cause/*`. Training never samples these pairs (`eval.heldout_pairs`).
- Baselines: random, oracle (ceiling), prompted LLM (`llm.HFPolicy`, `scripts/zero_shot.py`), human (Play page).
- W&B tracking (`tracking.py`): run name `<experiment>-<machine>-<sha>`, tags `machine:*`, `graph:*`; episodes logged
  as a table and a JSONL artifact. Use the same explicit `step` for `log_eval` and `log_episodes`.
- Streamlit analysis app (`app/`, `src/wikigame/webapp/`): Runs, Episodes (path graph, per-step "why invalid"), Failures,
  Diagnostics (invalid-cause chart, reward breakdown, flags), Baselines (Wilson CIs), Play. Reads W&B via a `DataSource`.
  `WIKIGAME_APP_SOURCE=fixture` for tests. Run: `uv run streamlit run app/streamlit_app.py`.

## Baselines so far (Wikispeedia, 50 pairs/hop, Qwen2.5-1.5B-Instruct, greedy, Mac)
| policy | h1 | h2 | h3 | h4 | overall |
|---|---|---|---|---|---|
| random | 6% | 0 | 0 | 0 | 1.5% |
| zero-shot strict | 88% | 36% | 10% | 0% | 33.5% |
| zero-shot lenient (accepts bare/```json titles) | 98% | 44% | 6% | 4% | 38.0% |
| strict + feedback | 94% | 44% | 12% | 4% | 38.5% |
| lenient + feedback | 98% | 56% | 10% | 4% | 42.0% |
| oracle | 100% | 100% | 100% | 100% | 100% |
Findings: ~60% of Qwen's actions are invalid (not navigation errors). Dominant cause is naming the target page even
when it is not linked (~40-47% of all actions), then re-picking the current page (~5-9%). Feedback helps only a little
(CIs overlap); the model mostly ignores it. n=10/hop numbers from early runs were too optimistic; trust the 50/hop ones.
R1 baseline: none on the Mac (by the user's choice); the untrained-R1 baseline was run later on the CUDA box at 20 pairs/hop (13.8%, see "CUDA box: state of play").

## Model decision: DeepSeek-R1-Distill-Qwen-1.5B (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`)
The user switched to this because Qwen-Instruct follows the format poorly. Handled in `llm.py`: system prompt (+ a
brevity hint) is folded into the user turn; the chat template ends the prompt with `<think>\n`, so the opening tag is
restored in stored output; sampling T=0.6 is recommended (greedy loops). Probe on the Mac (MPS): ~25 s per call,
~600-800 tokens per answer, ~half of answers cut off at 1024 tokens (it reasons about every link). Prompts are short
(median 307 tokens, max 792), so memory is dominated by completions.

## GRPO (per-step reward) - `scripts/grpo_train.py`, TRL 1.17.1 + PEFT LoRA
- One example = one game state (prompt). Model samples G answers; each is scored by replaying that move in the env
  (`grpo_reward.StepReward`). Dataset rows carry source/target/path/n_invalid so the state is rebuilt exactly.
- Training reward (shaping on; eval reward stays UNshaped): valid move +1 closer / 0 sideways / -1 farther (clipped to
  +-1; BFS distance to target), step cost -0.02, success +1, dead end -0.2. **Invalid answers are ranked** (user's
  decision, all below the worst valid move): real-but-illegal page -1.5 (not_linked, target_not_linked, current_page,
  revisit) < non-existent title -1.75 < unparseable -2.0 < cut off at the token cap with no `</answer>` -2.25.
  `--flat-invalid` = every invalid -2 (ablation). No warm start / SFT (user said no).
- States (`states.sample_states`): fresh starts, shortest-path prefixes, noisy epsilon-greedy walks (stand-in for the
  model's own mistakes), hops 1-3. 57% of sampled states are one move from the target: skewed easy, adjust `--hops/--mix`.
- Completion-cap curriculum (user's idea): stage 1 `--max-completion 1024`, then stage 2 `--max-completion 512` resuming
  the adapter with `--resume-adapter` (separate W&B runs in group `grpo-per-step`; eval uses a fixed cap in both).
  Resume restores adapter weights only, not optimizer state.
- Settings: G=8, 4 prompts/step, micro-batch 4, T=0.7, lr 1e-5, beta=0 (no ref model), TRL default loss `dapo`,
  group-normalised advantages. `--max-steps` also sizes the state dataset. If OOM: lower `--micro-batch` (must divide
  prompts-per-step x G). Metrics to watch: `sys/cuda_peak_gb`, `env/invalid_rate`, `env/closer_rate`,
  `env/cutoff_rate`, `env/invalid/<cause>`, `rewards/env_step_reward/mean`, `frac_reward_zero_std` (groups with no
  learning signal), `completions/clipped_ratio`.
- Evaluate a checkpoint on the shared held-out pairs (job_type `eval`, appears on the Baselines page):
  `uv run python scripts/zero_shot.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --adapter data/ckpt/<exp> --name <exp> --lenient --temperature 0.6 --max-new-tokens 1024 --n-per-hop 50`
- Checked only on the Mac with Qwen 0.5B/1.5B (fp32, MPS): loop runs, gradients flow, W&B (offline) gets env/* metrics
  and completion tables, adapter reloads for eval. **R1 has never been trained; GPU memory/speed are unmeasured.**
  Plain HF generation of ~700 tokens x 32 sequences per step will be slow (vLLM colocate mode is the upgrade path).
- Suggested first CUDA session: short dry run (`--max-steps 5`), check peak memory, `frac_reward_zero_std`, cutoff rate
  and step time; then stage 1; eval a checkpoint; ask the user before stage 2 and before any long run.

## CUDA box: state of play (written 2026-11-08/09 by the agent on the RTX machine, for the Mac agent)
**The "rented CUDA box" is a Windows 11 PC** (Git Bash + PowerShell; RTX 5070 Ti 16 GB, 64 GB RAM, host Gaming-Machine), not Linux.
Training and evals are FINISHED; results below. Hook and `detect_machine()` accept Windows + working `nvidia-smi` -> CUDA.

### Final results (held-out pairs, first 20 per hop of the 50-per-hop pool, 80 games/model, R1 base, `--lenient --temperature 0.6 --max-new-tokens 1024`)
| model | W&B run | overall | h1 | h2 | h3 | h4 | invalid | unparseable |
|---|---|---|---|---|---|---|---|---|
| untrained R1 (baseline) | `en686sia` | 13.8% | 50% | 5% | 0% | 0% | 73% | 68% |
| stage 1b final (cap 1024) | `zw66dml9` | 26.3% | 80% | 25% | 0% | 0% | 52% | 35% |
| stage 2 step 100 | `q98p2apa` | 33.8% | 85% | 45% | 5% | 0% | 38% | 11% |
| **stage 2 final (cap 512)** | `a7ldzv08` | **35.0%** | 80% | 40% | 20% | 0% | 37% | 5% |
- Training roughly 2.5x'd success and removed most unparseable output; stage 2 (shorter cap) beat stage 1b. Remaining invalid actions are mostly
  `unknown_title` (~13%, hallucinated page names), `target_not_linked` (6%), `revisit` (6%). No model solves any hop-4 game.
- **Caveats:** n=20/hop = wide CIs (one game = 1.25 points overall). The Mac's Qwen2.5-1.5B zero-shot (lenient, n=50/hop, greedy) scored 38%, so the
  trained R1 is on par, not clearly better; the settings differ (model, n, decoding), so do not over-read it. Per-step training success (47%) is
  far above full-game success (35%) because 57% of training states are one move from the target: train on harder states (`--hops/--mix`) next.
- Eval local copies: `data/runs/*.jsonl` (episodes + raw text), logs `data/logs/eval-*.log`; one eval took 0.7-1.7 h on the RTX.

### What was run (all with the defaults in the GRPO section: G=8, 4 prompts/step, lr 1e-5, LoRA r16, beta 0)
- Dry run (5 steps): ~85-105 s/step, peak VRAM 8.6 GB. Untrained R1 rambles to the token cap with no `<answer>` (invalid 94-97%), so the
  reward code was right and early learning is "finish and emit the tag".
- **Stage 1**: first attempt killed at step 32 (W&B `ceebhgej`; its `checkpoint-25` is the restart point), then **stage "1b"**: `--stage 1b --max-steps 275
  --seed 2 --resume-adapter data/ckpt/grpo-step-r1-stage1/checkpoint-25 --out data/ckpt/grpo-step-r1-stage1b` (W&B `kkqk572s`, 8h51m wall, ~85 s/step).
  Effective stage 1 = checkpoint-25 + stage 1b; optimizer state was not carried over (resume restores adapter weights only). Invalid rate
  96% -> 36%, cutoff 53% -> 12%, per-step success 2% -> 23%, completion length 838 -> ~520 tokens.
- **Stage 2**: `--stage 2 --max-completion 512 --max-steps 300 --seed 1 --resume-adapter data/ckpt/grpo-step-r1-stage1b --out data/ckpt/grpo-step-r1-stage2`
  (W&B `sa7pbhpc`, 4h35m, ~40-55 s/step). `--seed` differs per stage on purpose: the same seed re-samples the same states. End state: invalid 20%,
  cutoff 1%, per-step success 47%, ~179 tokens/completion, peak VRAM 9.0 GB.
- Checkpoints every 25 steps (adapters in `data/ckpt/<out>/checkpoint-<N>`); final adapters in the run dirs. `--save-total-limit` default 0 keeps all.
- W&B housekeeping (user-approved): deleted the crashed `k8mnksk6` and the 3-step aborted `pm3m0btf`; kept `ceebhgej` (steps 1-32, the only record of the
  slowdown) and left the Mac's early duplicates `henp9oon`, `pumln526`. A `handoff` run per phase also lives in the project.

### Windows / RTX lessons (read before running anything on this box)
- **WDDM VRAM spill = 4x slowdown (cost ~3 h).** A second GPU job during training made PyTorch's CUDA cache balloon to 15 GB dedicated + 4 GB spilled
  into shared system RAM (`Get-Counter '\GPU Process Memory(*)\Shared Usage'`) although `sys/cuda_peak_gb` stayed ~8.8; steps went 85 s -> 380-480 s with the GPU
  "99% busy" but only ~87 W of 300 W, no throttling, no recovery. **Never run a second GPU job while training or evaluating.** Fix: `grpo_train.py
  --gpu-mem-fraction` (default 0.75). `PYTORCH_CUDA_ALLOC_CONF=...max_split_size_mb` did NOT help.
- **Always set `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`** for training: TRL's `log_completions` prints sampled text and crashed the first launch with
  `UnicodeEncodeError` (cp1252). In ad-hoc scripts use `python -I -X utf8` (`-I` ignores `PYTHONUTF8`).
- **The PC can sleep**: it slept 20:11-20:43 on 2026-11-08 (something called `SetSuspendState`, although sleep/hibernate timers are "never" on AC). The eval froze and
  resumed intact, but wall-clock timings (e.g. s/call) jump. Check System log Kernel-Power events 42/107 when progress stalls; ask the user not to sleep the PC during long jobs.
- Long jobs run detached via PowerShell `Start-Process` (no tool timeout can kill them) with logs in `data/logs/`; unattended chains live in gitignored
  `data/run_stage2_after_stage1b.ps1` and `data/run_evals_after_stage2.ps1`. Stage 2 and the evals started themselves only if the previous step left its final adapter.
- torch: uv resolved a CPU-only wheel on Windows, so `pyproject.toml` pins the `pytorch-cu132` index for win32/linux only (Blackwell sm_120 needs CUDA >= 12.8);
  macOS keeps the PyPI wheel (MPS). `uv.lock` changed: `uv sync` on the Mac. No vLLM on Windows (plain HF generation). `tests/test_app_*.py` need `--group app`.
- **`--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` is required for adapter evals** (`zero_shot.py` defaults to Qwen2.5-1.5B).
- Observation, not set up by the agent: an `sshd` (OpenSSH Server) service is running on this PC (Automatic). The handoff does not use it; the user may disable it.
- `data/wikispeedia.json` was copied in by the user (rebuilding needs the raw SNAP files, not on this box).

## HANDOFF: CUDA machine -> dev machine (dev agent: poll `git pull` and grep for the `STATUS_` lines below)
STATUS_TRAINING: READY_FOR_TRANSFER  (2026-11-08 21:15 IST; artifact `aiwizards/wiki-game/wikigame-handoff-training:v0`, 56 files, 1833 MB)
STATUS_EVAL: PENDING  (evals are still running on the CUDA box; this line flips to READY_FOR_TRANSFER later, with its own artifact + manifest)
- **CHANGE (user, 2026-11-08 ~21:00): the handoff is now in two phases.** The training outputs are final and are ready NOW, so take
  them without waiting for the evals (which need several more hours). The eval data follows in phase 2.
- Transport: W&B artifacts, NOT scp (nothing is exposed on the Windows PC). File names inside an artifact are repo-relative, so
  downloading with `--root .` restores them in place. Each phase has its own manifest in git for integrity checking.
- **Phase 1 (training, ready):** every checkpoint's adapter files (`adapter_model.safetensors` + `adapter_config.json`, ~70 MB each,
  NO optimizer state) of `data/ckpt/grpo-step-r1-stage1b` (25..275), `data/ckpt/grpo-step-r1-stage2` (25..300), and
  `data/ckpt/grpo-step-r1-stage1/checkpoint-25` (the point stage 1b resumed from, so effective stage 1 = that + stage1b); the final adapter
  + `wikigame_config.json` of stage1b and stage2 (the root-level adapters equal checkpoint-275 / checkpoint-300); small logs in
  `data/logs`. **Model to test = `data/ckpt/grpo-step-r1-stage2`** (final). NOT shipped: optimizer states, `data/hf` (the Mac downloads
  the base model itself), stage logs > 20 MB (same metrics in W&B runs `kkqk572s` stage 1b and `sa7pbhpc` stage 2).
- Dev-agent steps for phase 1:
  1. `git pull`; `uv sync --group llm --group dev`.
  2. `uv run wandb artifact get aiwizards/wiki-game/wikigame-handoff-training:latest --root .`
  3. `uv run python scripts/publish_handoff.py --phase training --verify .`  must print `all 56 files verified OK` (size + sha256 vs
     `handoff/MANIFEST-training.txt`, which arrives through git). If it reports a problem, re-download; do not use the files.
  4. Smoke test on the Mac (MPS, ~25 s/call, so keep it tiny):
     `HF_HOME=data/hf uv run python scripts/zero_shot.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --adapter data/ckpt/grpo-step-r1-stage2 --lenient --temperature 0.6 --max-new-tokens 1024 --n-per-hop 3 --no-wandb --name mac-smoke`
- **`--model` is required for adapter evals.** `zero_shot.py`'s default model is Qwen2.5-1.5B, so an R1 LoRA would be applied to the
  wrong base model without it. Always pass the R1 id.
- **Phase 2 (eval, pending):** `data/runs/*.jsonl` (full eval episodes with raw model text) + eval logs, via artifact
  `wikigame-handoff-eval` and `handoff/MANIFEST-eval.txt`; same steps with `--phase eval`. Wait for `STATUS_EVAL: READY_FOR_TRANSFER`.
  The CUDA-side evals (unattended chain `data/run_evals_after_stage2.ps1`, log `data/logs/evalchain.log`) use 20 pairs/hop (the first 20 of
  the shared 50-per-hop held-out pool, hops 1-4, so 80 games each), R1 base, `--lenient --temperature 0.6 --max-new-tokens 1024`:
  `grpo-step-r1-stage2-final`, `grpo-step-r1-stage1b-final`, `zeroshot-r1-1.5b-lenient` (untrained R1 baseline),
  `grpo-step-r1-stage2-step100`. Results also appear as W&B runs (job_type `eval` / `baseline`) as each finishes. n=20/hop is small: wide CIs.
- Please avoid heavy GPU use on the CUDA box while its evals run; the dev machine is unaffected by any of this.

## Open items / ideas not done
- On-policy state rounds (retrain in rounds with states from the model's own rollouts); only the noisy-walk stand-in exists.
- R1 baseline and trained-model evals at 50/hop (done only at 20/hop so far, to save time); harder training states (hops 3-4) and on-policy rounds are the next levers.
- Simple English Wikipedia dump loader (bigger graph), TF-IDF/embedding baselines, trajectory-only reward ablation
  (separate W&B group), verifying the paper's table numbers (reference page figures came from a summary, not the PDF).
- Optional cleanup the user may want: duplicate early W&B runs `henp9oon`, `pumln526` (no episode tables); never delete
  anything without asking.
