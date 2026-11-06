"""Play-it-yourself support: build a game, and upload finished human games to W&B as a baseline run."""
from __future__ import annotations

import json

from ..env import WikiGameEnv
from ..eval import make_pairs, summarize
from ..graph import WikiGraph
from ..rollout import Episode, EpisodeRecorder


def human_text(title: str) -> str:
    """Human moves are recorded in the same <think>/<answer> format as agent output."""
    return f"<think>human</think><answer>{json.dumps({'title': title})}</answer>"


def pair_for(graph: WikiGraph, n_hops: int, game_no: int, held_out: bool, seed: int) -> tuple[int, int]:
    """Held-out mode walks the fixed eval pairs (same seed as scripts/smoke_run.py), so human results
    are directly comparable with the baselines; otherwise a fresh random pair."""
    if held_out:
        pairs = make_pairs(graph, hops=(n_hops,), n_per_hop=50)[n_hops]
        return pairs[game_no % len(pairs)]
    import random

    return graph.sample_pair(n_hops, random.Random(seed + game_no))


def new_game(graph: WikiGraph, n_hops: int, max_steps: int, pair: tuple[int, int], seed: int = 0):
    """Returns (env, recorder, observation_text)."""
    env = WikiGameEnv(graph, max_steps=max_steps, seed=seed)
    obs, info = env.reset(options={"source": pair[0], "target": pair[1]})
    return env, EpisodeRecorder(env, info, n_hops), obs


def upload_human_run(episodes: list[Episode], graph: WikiGraph, graph_name: str, max_steps: int):
    """Log finished human games as a W&B run (job_type=human, tag `human`). Returns the run URL.
    Only ever called from an explicit button press."""
    from ..tracking import init_run, log_episodes, log_eval

    run = init_run("human-baseline", "human",
                   {"policy": "human", "max_steps": max_steps, "n_games": len(episodes),
                    "hops": sorted({e.n_hops for e in episodes if e.n_hops is not None})},
                   graph, graph_name, tags=["human"])
    try:
        log_eval(run, summarize(episodes), step=0)
        log_episodes(run, episodes, step=0, name="eval")
        return run.url
    finally:
        run.finish()
