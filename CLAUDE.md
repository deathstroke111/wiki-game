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
**No R1 baseline exists yet** (only a 4-game probe), by the user's choice: "see how it performs after training".

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

## CUDA box notes (written 2026-11-08 by the agent on the RTX machine, for the Mac agent)
- **The "rented CUDA box" is a Windows 11 PC** (Git Bash + PowerShell; RTX 5070 Ti 16 GB, 64 GB RAM, host Gaming-Machine),
  not Linux. The hook and `detect_machine()` now accept Windows + working `nvidia-smi` -> CUDA (committed).
- torch: uv picked a CPU-only wheel on Windows, so `pyproject.toml` pins the `pytorch-cu132` index for win32/linux only
  (Blackwell sm_120 needs CUDA >= 12.8). macOS still resolves the PyPI wheel (MPS). `uv.lock` changed: `uv sync` on the Mac.
- No vLLM on Windows: generation is plain HF. `tests/test_app_*.py` need `--group app`; the other 40 tests pass here.
- Windows gotchas: print non-ASCII (R1 special tokens) with `python -I -X utf8` (`-I` ignores `PYTHONUTF8`);
  `data/wikispeedia.json` was copied in by the user (rebuild needs the raw SNAP files, not on this box).
- Power: sleep/hibernate are "never" on AC, so long runs are safe. Logs: `data/logs/*.log` (gitignored).
- **Dry run (5 steps, defaults: G=8, 4 prompts/step, micro-batch 4, cap 1024, R1-1.5B bf16 + LoRA r16):**
  ~85-105 s/step (so 300 steps ~ 7.5 h), peak VRAM 8.6 GB (lots of headroom: could raise `--micro-batch`), `frac_reward_zero_std`
  0, grad_norm ~0.06. Untrained R1: invalid 94-97% (unparseable ~90%), cutoff at 1024 tokens 44-50%, closer 0-3%.
  Sampled 8 completions by hand at one start state: all ramble ("Alternatively, maybe...") to the cap with no `<answer>`;
  the reward code scored them correctly, so the high invalid rate is model behaviour, not a plumbing bug.
  Early stage-1 learning is therefore mostly "finish and emit the tag"; watch `env/invalid_rate` and `env/cutoff_rate` fall first.
- `grpo_train.py` now has `--save-total-limit` (default 0 = keep ALL checkpoints, so each can be evaluated later).
  Checkpoints are LoRA adapters in `data/ckpt/<out>/checkpoint-<step>`; `zero_shot.py --adapter <dir>` loads them.
  Trainer-level resume is not exposed; `--resume-adapter` restores weights only (no optimizer state).
- **Always run training with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.** The first stage-1 launch crashed at step 1 with
  `UnicodeEncodeError` (cp1252 console) because TRL's `log_completions` prints sampled text (it contained 'ố'); the W&B-off
  dry run could not hit it. That aborted W&B run `k8mnksk6` (`...-cuda-2419725`, no useful data) is left in the project; don't delete without asking.
- **Stage 1 main run relaunched 2026-11-08 ~03:10 IST**, detached via PowerShell `Start-Process` (so no tool timeout can kill it;
  stdout/stderr -> `data/logs/stage1.log` / `stage1.err`): `--stage 1 --max-completion 1024 --max-steps 300 --save-steps 25
  --out data/ckpt/grpo-step-r1-stage1`. W&B run `ceebhgej` = `grpo-step-r1-1.5b-stage1-cuda-5762c82` (group grpo-per-step).
  Steps 1-2 at ~80-108 s/step => ~7-9 h total. Checkpoints land every 25 steps in `data/ckpt/grpo-step-r1-stage1/checkpoint-<N>`.

- **WDDM VRAM spill = 4x slowdown (found 2026-11-08, cost ~3 h).** At step ~10 of the first stage-1 run I started a second GPU
  job (a 1-step resume test) and the training process's CUDA cache ballooned to 15.3 GB dedicated + 4.2 GB spilled into shared
  system RAM (Get-Counter `\GPU Process Memory(*)`), even though `sys/cuda_peak_gb` stayed ~8.8. Windows then silently pages over
  PCIe: steps went 85 s -> 380-480 s, GPU 99% "utilized" but only ~87 W of 300 W, no throttling. It never recovered.
  **Never run a second GPU job while training. Diagnose with nvidia-smi power draw + Get-Counter shared usage.**
  Fix: `grpo_train.py --gpu-mem-fraction` (default 0.75, `torch.cuda.set_per_process_memory_fraction`) so PyTorch frees its cache
  instead; back to 83-93 s/step at 10.6 GB. (`PYTORCH_CUDA_ALLOC_CONF=...max_split_size_mb` did NOT help; don't use it.)
- **Run history:** first stage-1 run got to step 32 (W&B `ceebhgej`, killed as too slow; its `checkpoint-25` is the restart
  point). The relaunch with the allocator env var (W&B run from ~06:07) was also aborted after 3 steps. Current run =
  **stage "1b"**: `--stage 1b --max-steps 275 --seed 2 --resume-adapter data/ckpt/grpo-step-r1-stage1/checkpoint-25
  --out data/ckpt/grpo-step-r1-stage1b` (275 = 300 minus the 25 already done; seed 2 = fresh states), launcher pid 7780,
  logs `data/logs/stage1b.log/.err`. So effective stage 1 = checkpoint-25 of `stage1` + 275 steps of `stage1b`; optimizer state
  was not carried over (resume restores adapter weights only). **W&B cleanup (user-approved 2026-11-08):** deleted the crashed run
  `k8mnksk6` and the 3-step aborted relaunch `pm3m0btf`. Kept: `ceebhgej` (first stage-1 run, steps 1-32; source of checkpoint-25
  and the only record of the slowdown; shows state "running" because it was killed, not finished) and the LIVE stage-1b run
  `kkqk572s` (`...stage1b-cuda-02e16d3+dirty`, "+dirty" only because grpo_train.py was uncommitted at launch). Also left alone:
  the Mac's early duplicates `henp9oon`, `pumln526`.
- **Learning curve of stage 1b (20-step means, 2026-11-08 at step 162/275):** invalid rate 96% -> 80% (steps 61-80) -> 50% (101-120)
  -> 35% (141-162); cutoff 53% -> 13-16%; closer-move rate 2% -> 31%; per-step success 2% -> 22-30%; mean reward -2.04 -> -0.2;
  completion length 838 -> ~570 tokens. Step time steady ~85 s (wall ~106 s/step incl. checkpoint/W&B overhead).
  So the slow start was just that: the lr 1e-5 / LoRA r16 defaults do learn, no hyperparameter change needed so far.
- **Stage 2 is chained to start automatically (user approved 2026-11-08; evals only AFTER training).** Watcher
  `data/run_stage2_after_stage1b.ps1` (gitignored, powershell pid 21032) waits for the stage-1b launcher (pid 7780), and only if
  `data/ckpt/grpo-step-r1-stage1b/{wikigame_config.json,adapter_model.safetensors}` exist it runs stage 2:
  `--stage 2 --max-completion 512 --max-steps 300 --save-steps 25 --seed 1 --resume-adapter data/ckpt/grpo-step-r1-stage1b
  --out data/ckpt/grpo-step-r1-stage2`. `--seed 1` is deliberate: seed 0 would re-sample the exact same 1448 states stage 1
  used (it consumes ~1200). Progress: `data/logs/chain.log`, `stage2.log/.err`. If stage 1 dies, stage 2 is NOT started
  (chain.log says so); to continue from a checkpoint use `--resume-adapter data/ckpt/grpo-step-r1-stage1/checkpoint-<N>`.
  The resume path was smoke-tested (1 step from the dry-run adapter, 5.1 GB peak). Evals (zero_shot.py --adapter ...
  --lenient --temperature 0.6 --max-new-tokens 1024 --n-per-hop 50) are to run after both stages, one per checkpoint of interest.

- **Status 2026-11-08 17:30 IST:** stage 1b FINISHED cleanly (275/275 in 8h51m wall; final adapter + checkpoints 25..275 in
  `data/ckpt/grpo-step-r1-stage1b`). Last 35 steps: invalid 36%, cutoff 12%, closer 29%, per-step success 23%, reward -0.34.
  Stage 2 auto-started 15:08 (pid 7948) and is at step 143/300, ~40-56 s/step (512 cap is much faster; ETA ~2.5 h). Last 40 logged
  steps: invalid 20%, cutoff 1%, closer 53%, per-step success 47%, reward +0.55, mean length 179 tokens, peak VRAM 9.0 GB. No
  errors. Stage 2 checkpoints so far 25..125 in `data/ckpt/grpo-step-r1-stage2`. Evals not started (user: only after training).

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
- R1 baseline at 50/hop (ask the user; ~25 s/call on Mac, much faster on CUDA).
- Simple English Wikipedia dump loader (bigger graph), TF-IDF/embedding baselines, trajectory-only reward ablation
  (separate W&B group), verifying the paper's table numbers (reference page figures came from a summary, not the PDF).
- Optional cleanup the user may want: duplicate early W&B runs `henp9oon`, `pumln526` (no episode tables); never delete
  anything without asking.
