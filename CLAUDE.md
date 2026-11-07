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
  `uv run python scripts/zero_shot.py --adapter data/ckpt/<exp> --name <exp> --lenient --temperature 0.6 --max-new-tokens 1024 --n-per-hop 50`
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

- **Stage 2 is chained to start automatically (user approved 2026-11-08; evals only AFTER training).** Watcher
  `data/run_stage2_after_stage1.ps1` (gitignored, powershell pid 3276) waits for stage 1's launcher (pid 6280), and only if
  `data/ckpt/grpo-step-r1-stage1/{wikigame_config.json,adapter_model.safetensors}` exist it runs stage 2:
  `--stage 2 --max-completion 512 --max-steps 300 --save-steps 25 --seed 1 --resume-adapter data/ckpt/grpo-step-r1-stage1
  --out data/ckpt/grpo-step-r1-stage2`. `--seed 1` is deliberate: seed 0 would re-sample the exact same 1448 states stage 1
  used (it consumes ~1200). Progress: `data/logs/chain.log`, `stage2.log/.err`. If stage 1 dies, stage 2 is NOT started
  (chain.log says so); to continue from a checkpoint use `--resume-adapter data/ckpt/grpo-step-r1-stage1/checkpoint-<N>`.
  The resume path was smoke-tested (1 step from the dry-run adapter, 5.1 GB peak). Evals (zero_shot.py --adapter ...
  --lenient --temperature 0.6 --max-new-tokens 1024 --n-per-hop 50) are to run after both stages, one per checkpoint of interest.

## Open items / ideas not done
- On-policy state rounds (retrain in rounds with states from the model's own rollouts); only the noisy-walk stand-in exists.
- R1 baseline at 50/hop (ask the user; ~25 s/call on Mac, much faster on CUDA).
- Simple English Wikipedia dump loader (bigger graph), TF-IDF/embedding baselines, trajectory-only reward ablation
  (separate W&B group), verifying the paper's table numbers (reference page figures came from a summary, not the PDF).
- Optional cleanup the user may want: duplicate early W&B runs `henp9oon`, `pumln526` (no episode tables); never delete
  anything without asking.
