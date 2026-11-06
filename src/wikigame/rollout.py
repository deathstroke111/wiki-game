"""Single rollout code path shared by baselines, eval, and (later) the GRPO reward function.

A *policy* is any callable ``policy(observation: str, env: WikiGameEnv) -> str`` returning raw text in the
model format (``<think>..</think><answer>{"title": ...}</answer>``). LLM policies ignore ``env``; scripted
policies may peek at it. Emitting raw text keeps ``parse_action`` in the loop for every kind of policy.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Callable

from .env import WikiGameEnv, parse_action

Policy = Callable[[str, WikiGameEnv], str]


@dataclass
class Episode:
    source: str
    target: str
    optimal_steps: int
    n_hops: int | None  # curriculum level the pair was sampled at, if known
    path: list[str]  # titles actually visited, starting at source
    steps: list[dict] = field(default_factory=list)  # per-action: chosen, valid, reward, n_links, text
    success: bool = False
    extra_steps: int | None = None  # steps above optimal (successes only)
    n_valid: int = 0
    n_invalid: int = 0
    end_reason: str = ""  # success | dead_end | step_limit | invalid_action
    total_reward: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def run_episode(
    env: WikiGameEnv, policy: Policy, options: dict | None = None, n_hops: int | None = None
) -> Episode:
    obs, info = env.reset(options=options)
    g = env.graph
    ep = Episode(
        source=g.titles[info["source"]],
        target=g.titles[info["target"]],
        optimal_steps=info["optimal_steps"],
        n_hops=n_hops,
        path=[g.titles[info["source"]]],
    )
    terminated = truncated = False
    while not (terminated or truncated):
        n_links = len(env.links)
        text = policy(obs, env)
        title = parse_action(text)
        obs, reward, terminated, truncated, info = env.step(title)
        ep.total_reward += reward
        ep.steps.append(
            {"chosen": title, "valid": info["valid"], "reward": reward, "n_links": n_links, "text": text}
        )
        if info["valid"]:
            ep.n_valid += 1
            ep.path.append(g.titles[env.cur])
        else:
            ep.n_invalid += 1
    ep.success = bool(info.get("success", False))
    ep.extra_steps = info.get("extra_steps")
    ep.end_reason = "success" if ep.success else info.get("reason", "step_limit")
    return ep


# -- scripted policies (baselines / test oracles) ----------------------------------------------------


def _answer(title: str, think: str) -> str:
    return f"<think>{think}</think><answer>{json.dumps({'title': title})}</answer>"


def random_policy(seed: int = 0) -> Policy:
    """Uniformly random unvisited link. The floor every learned agent must beat."""
    rng = random.Random(seed)

    def policy(obs: str, env: WikiGameEnv) -> str:
        return _answer(rng.choice(env.links), "random") if env.links else ""

    return policy


def oracle_policy() -> Policy:
    """Always takes the link with the smallest true distance to the target. The ceiling (extra_steps == 0)."""
    cache: dict[int, dict[int, int]] = {}

    def policy(obs: str, env: WikiGameEnv) -> str:
        if env.target not in cache:
            cache.clear()
            cache[env.target] = env.graph.dist_to(env.target)
        d = cache[env.target]
        best = min(env.links, key=lambda t: d.get(env.graph.id_of(t), 10**9))
        return _answer(best, "oracle")

    return policy
