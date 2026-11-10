# wiki-game: teaching a small LLM to play the Wikipedia game with RL

Start on one Wikipedia article and reach a target article by clicking links. This repo trains a 1.5B reasoning model to do that with **GRPO**, on an offline link graph where the true shortest path from any page is known.

It contains the game environment, baselines, a per-step reward, the training script, a held-out evaluation, and a Streamlit app for analysing runs. It was inspired by ["Can Language Models Play the Wikipedia Game?"](https://arxiv.org/html/2610.13948v1), which benchmarks LLMs on the game without RL, and by `aigym`.

## Results (v0.1)

All agents play the same held-out games, 50 per hop, except where noted. "Hop *k*" means the shortest path is *k* clicks.

| Agent | Games | Overall | Hop 1 | Hop 2 | Hop 3 | Hop 4 |
|---|---|---|---|---|---|---|
| Random | 200 | 1.5% | 6% | 0% | 0% | 0% |
| Qwen2.5-1.5B zero-shot, greedy | 200 | 38.0% | 98% | 44% | 6% | 4% |
| Qwen2.5-1.5B zero-shot, T=0.6 | 200 | 42.0% | 98% | 50% | 14% | 6% |
| R1-Distill-Qwen-1.5B, untrained | 80 | 13.8% | 50% | 5% | 0% | 0% |
| R1-Distill-Qwen-1.5B, stage 2 | 80 | 35.0% | 80% | 40% | 20% | 0% |
| **R1-Distill-Qwen-1.5B, stage 3** | **200** | **45.0%** | **98%** | **54%** | **22%** | **6%** |

- On the 80 games it shares with the untrained model, stage 3 went from 13.8% to 47.5%. It won 27 games the untrained model lost and lost none that the untrained model won.
- About 77% of that gain comes from answering validly. More than one click from the target, move quality is only modestly above chance, and it matches chance at four or more clicks.
- Stage 3 is on par with an untrained Qwen at equal temperature: +3 points, p = 0.39.

The full analysis, with charts and caveats, is in [docs/findings/stage-3-findings.md](docs/findings/stage-3-findings.md).

## Quickstart

You need [uv](https://docs.astral.sh/uv/). `uv sync` removes any dependency group you don't pass, so always list the ones you want.

```bash
git clone https://github.com/deathstroke111/wiki-game.git && cd wiki-game
uv sync --group llm --group dev --group app
uv run pytest -q
```

Build the graph from the [Wikispeedia dataset](https://snap.stanford.edu/data/wikispeedia.html). It is not redistributed here.

```bash
mkdir -p data/raw/wikispeedia_graph data/raw/wikispeedia_text
curl -L https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz | tar -xz -C data/raw/wikispeedia_graph
curl -L https://snap.stanford.edu/data/wikispeedia/wikispeedia_articles_plaintext.tar.gz | tar -xz -C data/raw/wikispeedia_text
uv run python -I -m wikigame.data wikispeedia    # writes data/wikispeedia.json: 4,604 articles, 119,772 links
```

Then play one game in Python:

```python
from wikigame import WikiGameEnv, WikiGraph
from wikigame.rollout import oracle_policy, run_episode

env = WikiGameEnv(WikiGraph.load("data/wikispeedia.json"), seed=0)
episode = run_episode(env, oracle_policy())
print(episode.path, episode.success)
```

Or look around the analysis app with synthetic data, without any account:

```bash
WIKIGAME_APP_SOURCE=fixture uv run streamlit run app/streamlit_app.py
```

### Weights & Biases

Runs log to W&B unless you pass `--no-wandb`, and the app reads from W&B. The defaults point at the original authors' workspace, so use your own:

```bash
wandb login
export WANDB_ENTITY=<your-entity> WANDB_PROJECT=wiki-game    # or WANDB_MODE=offline
uv run python scripts/smoke_run.py                           # random and oracle baselines
uv run streamlit run app/streamlit_app.py
```

## How the code fits together

```text
Wikispeedia dump -> WikiGraph -> WikiGameEnv -> run_episode(policy) -> evaluate() on held-out pairs -> W&B -> Streamlit app
                                     ^                  ^
                     states.py samples positions        random / oracle / HFPolicy (Qwen, R1, R1 + adapter)
                     grpo_reward.py scores answers
                     scripts/grpo_train.py (TRL + LoRA)
```

If you want to learn from the code, this order follows the data:

| Read | What it teaches |
|---|---|
| [`src/wikigame/graph.py`](src/wikigame/graph.py) | The link graph and shortest-path distances |
| [`src/wikigame/env.py`](src/wikigame/env.py) | A Gymnasium-style environment: rules, action parsing, invalid-answer causes, reward config |
| [`src/wikigame/rollout.py`](src/wikigame/rollout.py) | The single game loop used everywhere, plus the random and oracle policies |
| [`src/wikigame/eval.py`](src/wikigame/eval.py) | Fixed held-out pairs and the success metrics |
| [`src/wikigame/llm.py`](src/wikigame/llm.py) | Prompts and PyTorch/Transformers generation, including LoRA adapters |
| [`src/wikigame/states.py`](src/wikigame/states.py) | The game positions used as training examples |
| [`src/wikigame/grpo_reward.py`](src/wikigame/grpo_reward.py) | Scoring each sampled answer by replaying that move |
| [`scripts/grpo_train.py`](scripts/grpo_train.py) | The TRL GRPO + PEFT LoRA training setup |
| [`src/wikigame/tracking.py`](src/wikigame/tracking.py) | Run naming, metrics and episode logging in W&B |
| [`src/wikigame/webapp/stats.py`](src/wikigame/webapp/stats.py) | Paired tests, bootstrap intervals and failure analysis |

Everything else:

- [`app/`](app/) holds the Streamlit pages: Runs, Episodes, Failures, Diagnostics, Baselines, Statistics and Play. `src/wikigame/webapp/` is its data and analysis layer.
- [`scripts/`](scripts/) holds the entry points. `zero_shot.py` evaluates a model or adapter, `probe.py` makes several model configs answer identical real positions, and `make_findings_figures.py` rebuilds the report's charts (it needs read access to the original W&B runs).
- [`docs/decisions/`](docs/decisions/README.md) records why each design choice was made, as ADRs. [`docs/LESSONS.md`](docs/LESSONS.md) collects the cross-cutting lessons.

## The game and the reward

Each turn the model sees the target's summary, the current page, the steps left and up to 200 link titles, then replies:

```text
<think>reasoning</think><answer>{"title": "Next Article"}</answer>
```

A game ends after 10 valid moves or 3 invalid answers. An invalid answer also ends the game once valid moves plus invalid answers reach 10. Pages can't be revisited, and a page with no unvisited links ends the game.

One GRPO example is one game position. The model samples 8 answers, each answer is scored by replaying that move, and answers that beat their group's average are reinforced.

| Answer | Training reward |
|---|---|
| Valid move: closer / same distance / farther | +1 / 0 / -1 |
| Reaches the target | additional +1 |
| Every answer | -0.02 |
| Dead end | -0.2 |
| Invalid: real page but illegal move | -1.5 |
| Invalid: title does not exist | -1.75 |
| Invalid: unparseable | -2.0 |
| Invalid: cut off at the token cap | -2.25 |

Invalid answers are ranked because a group of equally bad answers has no variance, so GRPO has nothing to learn from it. `--flat-invalid` turns the ranking off.

Held-out evaluation uses a separate, unshaped reward: +1 for success, -0.02 per answer and -0.1 per invalid answer.

## Training and evaluation

Check the plumbing on any machine with a tiny model. The numbers from this run mean nothing.

```bash
HF_HOME=data/hf uv run python scripts/grpo_train.py --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 3 \
    --num-generations 4 --prompts-per-step 1 --micro-batch 4 --max-completion 128 --no-wandb
```

Real training needs a GPU. Stage 3 peaked at 8.5 GB of VRAM and took about 34 s per step on a 24 GB RTX PRO 4000 Blackwell. `pyproject.toml` pins a CUDA 13.2 PyTorch build on Linux and Windows, so change that index if your driver is older.

```bash
export HF_HOME=data/hf
uv run python scripts/grpo_train.py --stage 1 --max-completion 1024 --max-steps 300
uv run python scripts/grpo_train.py --stage 2 --max-completion 512 --max-steps 300 --seed 1 \
    --resume-adapter data/ckpt/grpo-step-r1-stage1
uv run python scripts/grpo_train.py --stage 3 --max-completion 512 --max-steps 300 --seed 3 --save-steps 25 \
    --hops 2 3 4 --mix 0.6 0.2 0.2 --resume-adapter data/ckpt/grpo-step-r1-stage2
```

The original stage 1 crashed at step 32 and resumed from checkpoint 25 for another 275 steps, so an uninterrupted rerun will not match it exactly. Use a different `--seed` for each stage, because the same seed samples the same positions.

Evaluate an adapter on the same held-out games as every baseline. `--model` must match the adapter's base model.

```bash
uv run python scripts/zero_shot.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --adapter data/ckpt/grpo-step-r1-stage3 --name grpo-step-r1-stage3 \
    --lenient --temperature 0.6 --max-new-tokens 1024 --n-per-hop 50
```

A 200-game evaluation took about 135 minutes on that GPU, because games are currently played one at a time. Episodes are saved as JSONL under `data/runs/`.

## What is not in this repo

- **Trained adapters, datasets and logs.** `data/` is gitignored, so you train your own with the commands above.
- **The sampled Qwen run (T=0.6).** It is reported in the findings but was run locally and is not on W&B. To repeat it, run `zero_shot.py` with `--lenient --temperature 0.6`.
- **Scale.** Results are for the 4,604-article Wikispeedia graph and may not carry over to full Wikipedia. Each agent played each held-out game once, and 200 games can only reliably detect a change of about 10 points.
