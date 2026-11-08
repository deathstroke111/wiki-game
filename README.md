# wiki-game: RL for the Wikipedia game

Offline Wikipedia-game environment + evaluation + W&B tracking + an analysis webapp. Inspired by
["Can Language Models Play the Wikipedia Game?"](https://arxiv.org/html/2610.13948v1) and `aigym`.

Always use the project venv (`uv run ...`). Two machines: a local M4 Mac (dev/smoke) and a rented CUDA box (training);
`.claude/hooks/detect-machine.sh` reports which one a Claude session is on.

## Quickstart
```bash
uv sync --group app                                   # add --group dev for tests
uv run python -I -m wikigame.data wikispeedia         # build data/wikispeedia.json (needs data/raw/, see wikigame/data)
uv run pytest
wandb login                                           # once per machine (team: aiwizards)
uv run python scripts/smoke_run.py                    # random + oracle baselines -> W&B aiwizards/wiki-game
uv run streamlit run app/streamlit_app.py             # analysis webapp (reads W&B)
```
`WANDB_MODE=offline` + `wandb sync` works when the rented box has no network.

## Webapp pages (`app/views/`)
Runs (compare runs, curves) · Episodes (agent vs optimal path, graph view) · Failures (drift, wrong turns) ·
Diagnostics (health flags, reward breakdown) · Baselines (random/oracle/human/model per hop) · Play (you play; optional upload as a human baseline).

## Layout
- `src/wikigame/` graph, env, rollout, eval, tracking; `webapp/` data layer, analysis, plots for the app
- `app/` Streamlit pages · `scripts/` runnable entry points · `tests/`

## GRPO training (per-step reward)

One training example is one game state; the model samples G answers and each is scored by taking that move in the
env (`grpo_reward.StepReward`: +1 closer / 0 sideways / -1 farther, step cost, success bonus; invalid answers are ranked: real page but not a legal move -1.5 < non-existent title -1.75
< unparseable -2.0 < cut off at the token cap -2.25; `--flat-invalid` gives every invalid answer -2 for an ablation).
States come from `states.sample_states` (fresh starts, shortest-path prefixes, noisy walks; never the eval pairs).

```bash
# plumbing check on the Mac (tiny model; numbers are meaningless)
HF_HOME=data/hf uv run python scripts/grpo_train.py --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 3 \
    --num-generations 4 --prompts-per-step 1 --micro-batch 4 --max-completion 128 --no-wandb
# real run on the RTX: stage 1 (1024-token cap), then stage 2 resumes the adapter with a 512 cap
uv run python scripts/grpo_train.py --stage 1 --max-completion 1024
uv run python scripts/grpo_train.py --stage 2 --max-completion 512 --resume-adapter data/ckpt/grpo-step-r1-stage1
# evaluate a checkpoint on the same held-out pairs as every baseline
uv run python scripts/zero_shot.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --adapter data/ckpt/grpo-step-r1-stage1 --name grpo-step-r1-stage1 --lenient --temperature 0.6 --n-per-hop 50 --max-new-tokens 1024
```
