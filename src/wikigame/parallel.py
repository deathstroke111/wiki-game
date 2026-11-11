"""Play many games at once: one batched model call advances every running game by one move.

The model sits behind ``generate(list of chat messages) -> list of raw texts``, so the game logic is testable with a
scripted stub and the same player serves both evals (``HFPolicy.generate_batch``) and on-policy training rounds
(the model being trained, see ``onpolicy.py``). Games are scheduled like continuous batching: up to ``batch_size``
games run at a time and a finished game's slot goes to the next waiting game, so batches stay full and memory is
bounded by ``batch_size``, not by the number of games.

Besides the finished episodes, every position a game reached (the prompt the model answered there) is returned as a
training-state row in the ``states.sample_states`` format, which is how on-policy rounds feed GRPO.
"""
from __future__ import annotations

from typing import Callable

from .env import WikiGameEnv, parse_action
from .graph import WikiGraph
from .llm import build_messages, lenient_answer
from .rollout import Episode, EpisodeRecorder

BatchGenerate = Callable[[list[list[dict]]], list[str]]


def play_games(graph: WikiGraph, generate: BatchGenerate, games: list[tuple[int, int, int | None]], *,
               messages_fn: Callable[[str], list[dict]] = build_messages, batch_size: int = 16, max_steps: int = 10,
               feedback: bool = False, lenient: bool = False, seed: int = 0,
               on_episode: Callable[[int, Episode], None] | None = None) -> tuple[list[Episode], list[dict]]:
    """Play ``games`` = [(source, target, n_hops), ...] to the end. Returns (episodes in input order, state rows).

    ``messages_fn`` turns an observation into the chat messages sent to ``generate`` (and stored as the row's
    prompt). ``lenient`` applies ``lenient_answer`` to raw texts (leave it off when ``generate`` already does).
    Game i uses its own env seeded ``seed + i``. ``on_episode(i, episode)`` fires as each game ends."""
    n = len(games)
    envs: list[WikiGameEnv | None] = [None] * n
    recs: list[EpisodeRecorder | None] = [None] * n
    obs: list[str] = [""] * n
    episodes: list[Episode | None] = [None] * n
    rows: list[dict] = []
    waiting, running = list(range(n)), []
    while waiting or running:
        while waiting and len(running) < batch_size:
            i = waiting.pop(0)
            s, t, k = games[i]
            env = envs[i] = WikiGameEnv(graph, max_steps=max_steps, feedback=feedback, seed=seed + i)
            obs[i], info = env.reset(options={"source": s, "target": t})
            recs[i] = EpisodeRecorder(env, info, k)
            running.append(i)
        batch = [messages_fn(obs[i]) for i in running]
        for i, msgs in zip(running, batch):
            env = envs[i]
            rows.append({"prompt": msgs, "source": env.source, "target": env.target, "path": list(env.path),
                         "n_invalid": env.invalid, "kind": "onpolicy", "n_hops": games[i][2], "links": list(env.links),
                         "game": i})
        texts = generate(batch)
        if len(texts) != len(running):
            raise ValueError(f"generate returned {len(texts)} texts for {len(running)} prompts")
        still = []
        for i, text in zip(running, texts):
            text = lenient_answer(text) if lenient else text
            title = parse_action(text)
            obs[i], reward, terminated, truncated, info = envs[i].step(title)
            recs[i].record(text, title, reward, info)
            if terminated or truncated:
                episodes[i] = recs[i].finish()
                if on_episode:
                    on_episode(i, episodes[i])
            else:
                still.append(i)
        running = still
    return episodes, rows
