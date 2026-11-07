"""Training-state sampler for per-step GRPO.

One training example = one game state: the prompt the model sees there. The model's sampled answers are scored
by taking that move in the env (see grpo_reward.StepReward), so a dataset row also carries everything needed to
rebuild the state: source, target, path (node ids), n_invalid.

State kinds:
  start    the first move of a fresh game (curriculum: pairs at 1..k hops)
  optimal  a state partway along a shortest path (the model has done well so far)
  noisy    a state reached by a scripted epsilon-greedy walk (oracle moves with random wrong turns), a stand-in
           for the model's own mistakes until on-policy rollouts are wired in
Eval pairs (eval.heldout_pairs) are never sampled.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .env import WikiGameEnv
from .eval import heldout_pairs
from .graph import WikiGraph
from .llm import training_messages


@dataclass
class StateMix:
    start: float = 0.4
    optimal: float = 0.3
    noisy: float = 0.3
    hops: tuple[int, ...] = (1, 2, 3)
    epsilon: float = 0.35  # chance a noisy-walk move is random instead of the oracle's
    p_post_invalid: float = 0.0  # share of rows that are the state right after an invalid answer (needs feedback)


def _noisy_prefix(graph: WikiGraph, env: WikiGameEnv, dist: dict[int, int], s: int, t: int, rng: random.Random,
                  eps: float) -> list[int] | None:
    """Walk from s with epsilon-greedy moves; return the path if it ends in a playable, target-reachable state."""
    path, visited = [s], {s}
    for _ in range(rng.randint(1, max(1, env.max_steps - 3))):
        cand = [v for v in graph.adj[path[-1]] if v not in visited]
        if not cand:
            return None
        v = rng.choice(cand) if rng.random() < eps else min(cand, key=lambda c: dist.get(c, 10**9))
        if v == t:
            break
        path.append(v)
        visited.add(v)
    ok = len(path) > 1 and path[-1] != t and path[-1] in dist
    return path if ok and [v for v in graph.adj[path[-1]] if v not in visited] else None


def sample_states(graph: WikiGraph, n: int, model_id: str, mix: StateMix | None = None, seed: int = 0,
                  env: WikiGameEnv | None = None) -> list[dict]:
    """``n`` distinct training rows (columns: prompt, source, target, path, n_invalid, kind, n_hops, links)."""
    mix = mix or StateMix()
    rng = random.Random(seed)
    env = env or WikiGameEnv(graph, seed=seed)
    held = heldout_pairs(graph)
    kinds, weights = ["start", "optimal", "noisy"], [mix.start, mix.optimal, mix.noisy]
    rows, seen, tries = [], set(), 0
    while len(rows) < n and tries < n * 50:
        tries += 1
        k = rng.choice(mix.hops)
        s, t = graph.sample_pair(k, rng)
        if (s, t) in held:
            continue
        kind = rng.choices(kinds, weights)[0]
        path = [s]
        if kind == "optimal" and k >= 2:
            sp = graph.shortest_path(s, t)
            path = sp[: rng.randint(1, len(sp) - 2) + 1]
        elif kind == "noisy":
            path = _noisy_prefix(graph, env, graph.dist_to(t), s, t, rng, mix.epsilon)
            if path is None:
                continue
        else:
            kind = "start"
        last = None
        if env.feedback and rng.random() < mix.p_post_invalid:
            adjacent = t in graph.adj[path[-1]]
            last = ("current_page", graph.titles[path[-1]]) if adjacent else ("target_not_linked", graph.titles[t])
        key = (s, t, tuple(path), last is not None)
        if key in seen:
            continue
        seen.add(key)
        obs = env.set_state(s, t, path, n_invalid=1 if last else 0, last_invalid=last)
        rows.append({"prompt": training_messages(obs, model_id), "source": s, "target": t, "path": path,
                     "n_invalid": 1 if last else 0, "kind": kind, "n_hops": k, "links": list(env.links)})
    return rows
