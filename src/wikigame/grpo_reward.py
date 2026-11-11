"""GRPO reward function: score each sampled answer by taking that move in the env from the prompt's state.

TRL calls ``reward(prompts=..., completions=..., completion_ids=..., **dataset_columns)`` once per batch; every
dataset row carries source/target/path/n_invalid so the exact state is rebuilt with ``WikiGameEnv.set_state``.
The reward is the env's own step reward with shaping on (see RewardConfig): +1 closer / 0 sideways / -1 farther,
step cost, success bonus, dead-end penalty, and a ranked penalty for invalid answers (``invalid_by_cause``:
real-but-illegal page -1.5 < non-existent title -1.75 < unparseable -2.0), plus -0.25 more if the answer was cut
off at the token cap.
"""
from __future__ import annotations

from collections import Counter

from .env import RewardConfig, WikiGameEnv, parse_action
from .graph import WikiGraph


def default_reward_config() -> RewardConfig:
    return RewardConfig(shaping=1.0)  # eval reward stays unshaped (RewardConfig()); only training shapes


class StepReward:
    __name__ = "env_step_reward"  # TRL names the logged metric after this

    def __init__(self, graph: WikiGraph, reward: RewardConfig | None = None, max_completion_length: int | None = None,
                 max_steps: int = 10, max_invalid: int = 3, feedback: bool = False, cutoff_penalty: float = -0.25):
        self.env = WikiGameEnv(graph, max_steps=max_steps, max_invalid=max_invalid, reward=reward or default_reward_config(),
                               feedback=feedback, seed=0)
        self.max_completion_length = max_completion_length
        self.cutoff_penalty = cutoff_penalty  # extra on top of the unparseable tier: ran out of tokens, never answered

    @staticmethod
    def completion_text(c) -> str:
        return c[0]["content"] if isinstance(c, list) else c  # conversational: [{"role": "assistant", "content": ...}]

    def score(self, text: str, source: int, target: int, path: list[int], n_invalid: int) -> tuple[float, dict]:
        env = self.env
        env.set_state(source, target, list(path), n_invalid=n_invalid)
        state_dist = env._dist.get(env.cur) if env._dist is not None else None  # clicks to the target from this state
        _, reward, _, _, info = env.step(parse_action(text))
        info["state_dist"] = state_dist  # set for invalid answers too, so rates can be split by distance
        return reward, info

    def __call__(self, prompts, completions, source, target, path, n_invalid, completion_ids=None,
                 log_metric=None, **_):
        rewards, infos, cut = [], [], 0
        for i, c in enumerate(completions):
            text = self.completion_text(c)
            r, info = self.score(text, source[i], target[i], path[i], n_invalid[i])
            if completion_ids is not None and self.max_completion_length and "</answer>" not in text \
                    and len(completion_ids[i]) >= self.max_completion_length:
                cut += 1  # ran out of tokens before answering: a thinking-budget failure, not a format one
                r += self.cutoff_penalty
                info["cutoff"] = True
            rewards.append(r)
            infos.append(info)
        if log_metric:
            n = max(1, len(infos))
            log_metric("env/invalid_rate", sum(not x["valid"] for x in infos) / n)
            log_metric("env/closer_rate", sum(x.get("progress", 0) > 0 for x in infos) / n)
            log_metric("env/sideways_rate", sum(x["valid"] and x.get("progress") == 0 for x in infos) / n)
            log_metric("env/out_of_reach_rate", sum(bool(x.get("out_of_reach")) for x in infos) / n)
            log_metric("env/success_rate", sum(bool(x.get("success")) for x in infos) / n)
            log_metric("env/cutoff_rate", cut / n)
            for cause, k in Counter(x["cause"] for x in infos if "cause" in x).items():
                log_metric(f"env/invalid/{cause}", k / n)
            # The same rates split by clicks to the target (1, 2, 3, 4+), so a harder state mix is not mistaken
            # for worse play. A bucket is logged only for batches that contain it.
            by_dist: dict[str, list[dict]] = {}
            for x in infos:
                if x.get("state_dist") is not None:
                    by_dist.setdefault(f"d{min(x['state_dist'], 4)}{'p' if x['state_dist'] >= 4 else ''}", []).append(x)
            for b, xs in by_dist.items():
                log_metric(f"env/{b}/invalid_rate", sum(not x["valid"] for x in xs) / len(xs))
                log_metric(f"env/{b}/closer_rate", sum(x.get("progress", 0) > 0 for x in xs) / len(xs))
                log_metric(f"env/{b}/share", len(xs) / n)
        return rewards
