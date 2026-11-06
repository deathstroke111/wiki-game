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
