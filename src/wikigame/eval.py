"""Evaluation on fixed, held-out start/target pairs, bucketed by hop distance."""
from __future__ import annotations

import random
from statistics import mean

from .env import WikiGameEnv
from .graph import WikiGraph
from .rollout import Episode, Policy, run_episode

Pairs = dict[int, list[tuple[int, int]]]


def make_pairs(graph: WikiGraph, hops=(1, 2, 3, 4), n_per_hop: int = 100, seed: int = 1234) -> Pairs:
    """Deterministic eval pairs. Use a seed distinct from training so pairs stay held out."""
    rng = random.Random(seed)
    return {k: [graph.sample_pair(k, rng) for _ in range(n_per_hop)] for k in hops}


def summarize(episodes: list[Episode]) -> dict[str, float]:
    """Metrics from a list of episodes: success@k, extra_steps@k (successes only), plus overall rates."""
    m: dict[str, float] = {}
    for k in sorted({e.n_hops for e in episodes if e.n_hops is not None}):
        eps = [e for e in episodes if e.n_hops == k]
        m[f"n@{k}"] = len(eps)  # episodes behind success@k: lets the app draw honest error bars
        m[f"success@{k}"] = mean(e.success for e in eps)
        extras = [e.extra_steps for e in eps if e.success]
        if extras:
            m[f"extra_steps@{k}"] = mean(extras)
    actions = sum(e.n_valid + e.n_invalid for e in episodes)
    m["success"] = mean(e.success for e in episodes)
    m["invalid_rate"] = sum(e.n_invalid for e in episodes) / max(1, actions)
    m["path_len"] = mean(len(e.path) - 1 for e in episodes)
    m["reward"] = mean(e.total_reward for e in episodes)
    m["n_episodes"] = len(episodes)
    return m


def evaluate(env: WikiGameEnv, policy: Policy, pairs: Pairs) -> tuple[dict[str, float], list[Episode]]:
    episodes = [
        run_episode(env, policy, options={"source": s, "target": t}, n_hops=k)
        for k, plist in pairs.items()
        for s, t in plist
    ]
    return summarize(episodes), episodes
