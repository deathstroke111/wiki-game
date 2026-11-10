"""Replay real mid-game states and score single answers: lets several model configs face identical states."""
from __future__ import annotations

import random

from .env import classify_invalid, parse_action
from .graph import WikiGraph


def collect_states(graph: WikiGraph, episodes: list[dict]) -> list[dict]:
    """Every distinct state an agent answered in: the path so far and the BFS distance to the target."""
    seen: dict[tuple, dict] = {}
    distances: dict[str, dict[int, int]] = {}
    for ep in episodes:
        target = graph.id_of(ep["target"])
        if target is None:
            continue
        dist = distances.setdefault(ep["target"], graph.dist_to(target))
        path = [ep["path"][0]]
        for step in ep["steps"]:
            d = dist.get(graph.id_of(path[-1]))
            if d is not None:
                seen.setdefault((ep["target"], tuple(path)), {"target": ep["target"], "path": list(path), "dist": d})
            if step["valid"]:
                path.append(step["chosen"])
    return list(seen.values())


def sample_states(states: list[dict], dists: list[int], per_dist: int, seed: int = 0) -> list[dict]:
    """A reproducible sample of up to per_dist states at each distance (the last distance includes farther ones)."""
    rng = random.Random(seed)
    top = max(dists)
    out = []
    for d in dists:
        pool = [s for s in states if min(s["dist"], top) == d]
        out += rng.sample(pool, min(per_dist, len(pool)))
    return out


def score_answer(graph: WikiGraph, state: dict, text: str) -> dict:
    """What this answer would do from the state: closer, sideways, farther, or invalid with its cause."""
    target = graph.id_of(state["target"])
    ids = [graph.id_of(t) for t in state["path"]]
    action = parse_action(text)
    cause = classify_invalid(graph, ids[-1], set(ids), target, action)
    if cause is not None:
        return {"chosen": action, "outcome": "invalid", "cause": cause}
    dist = graph.dist_to(target)
    before, after = dist.get(ids[-1]), dist.get(graph.id_of(action))
    if before is None or after is None or after > before:
        outcome = "farther"
    else:
        outcome = "closer" if after < before else "sideways"
    return {"chosen": action, "outcome": outcome, "cause": None}
