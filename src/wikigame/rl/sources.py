"""Where training groups come from: prompts in, G scored answers per prompt out.

``StepSource`` is the per-step setup every stage so far used: one prompt = one game state, its G answers are scored
by replaying that one move in the env (``grpo_reward.StepReward``, unchanged). States come from a ``StatePool``
(scripted states, plus the latest on-policy round when ``OnPolicyRounds`` is on). A whole-game source (v2) will plug
in here with the same ``next_batch`` contract, without touching the algorithms or the learner.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from ..eval import heldout_pairs, summarize
from ..graph import WikiGraph
from ..grpo_reward import StepReward
from ..llm import training_messages
from ..onpolicy import StatePool, round_pairs, select_states
from ..parallel import play_games
from .generation import Generator, Sample, chat_batch_generate, encode_prompt, restore_think


@dataclass
class Group:
    row: dict  # the state (source, target, path, n_invalid, kind, ...)
    prompt_ids: list[int]
    samples: list[Sample]
    texts: list[str]  # answer texts as scored (<think> restored)
    rewards: list[float]


class StepSource:
    def __init__(self, pool: StatePool, tok, generator: Generator, reward: StepReward, *, num_generations: int,
                 max_tokens: int, temperature: float):
        self.pool, self.tok, self.gen, self.reward = pool, tok, generator, reward
        self.G, self.max_tokens, self.temperature = num_generations, max_tokens, temperature
        self.cursor = 0

    def next_batch(self, n_prompts: int) -> tuple[list[Group], dict[str, float]]:
        rows = [self.pool[self.cursor + i] for i in range(n_prompts)]
        self.cursor += n_prompts
        enc = [encode_prompt(self.tok, r["prompt"]) for r in rows]
        t0 = time.time()
        out = self.gen.generate([ids for ids, _ in enc], self.G, self.max_tokens, self.temperature)
        gen_s = time.time() - t0
        # score the whole batch in one call, as TRL did, so the env/* rates (incl. per-distance) are batch rates
        flat = [(row, s, restore_think(s.text, opened)) for row, (_, opened), group in zip(rows, enc, out) for s in group]
        metrics: dict[str, float] = {}
        rewards = self.reward(prompts=None, completions=[t for _, _, t in flat], source=[r["source"] for r, _, _ in flat],
                              target=[r["target"] for r, _, _ in flat], path=[r["path"] for r, _, _ in flat],
                              n_invalid=[r["n_invalid"] for r, _, _ in flat],
                              completion_ids=[s.token_ids for _, s, _ in flat], log_metric=metrics.__setitem__)
        groups, i = [], 0
        for row, (ids, _), samples in zip(rows, enc, out):
            n = len(samples)
            groups.append(Group(row, ids, samples, [t for _, _, t in flat[i:i + n]], rewards[i:i + n]))
            i += n
        metrics["time/generate_s"] = gen_s
        metrics["onpolicy/drawn_this_step"] = sum(r["kind"] == "onpolicy" for r in rows) / len(rows)
        return groups, metrics


class OnPolicyRounds:
    """Every ``every`` steps (and before step 1) the current model plays ``n_games`` whole games on fresh training
    pairs; the positions it reached replace the pool's on-policy states (same as stage 4's TRL callback)."""

    def __init__(self, pool: StatePool, graph: WikiGraph, generator: Generator, tok, *, every: int, n_games: int,
                 hops: tuple[int, ...], model_id: str, hint: str | None, max_steps_game: int, max_tokens: int,
                 temperature: float, batch_size: int, feedback: bool, seed: int, log_path: Path | None = None):
        self.pool, self.graph, self.gen, self.tok = pool, graph, generator, tok
        self.every, self.n_games, self.hops = every, n_games, hops
        self.model_id, self.hint = model_id, hint
        self.max_steps_game, self.max_tokens, self.temperature = max_steps_game, max_tokens, temperature
        self.batch_size, self.feedback, self.seed, self.log_path = batch_size, feedback, seed, log_path
        self.held = heldout_pairs(graph)

    def due(self, step: int) -> bool:
        return bool(self.every) and step % self.every == 0

    def play(self, step: int) -> dict[str, float]:
        t0 = time.time()
        rng = random.Random(self.seed * 100_003 + step)
        games = round_pairs(self.graph, self.n_games, self.hops, rng, self.held)
        generate = chat_batch_generate(self.gen, self.tok, self.max_tokens, self.temperature)
        episodes, rows = play_games(self.graph, generate, games, batch_size=self.batch_size,
                                    max_steps=self.max_steps_game, feedback=self.feedback, lenient=True,
                                    seed=rng.randrange(2**31),
                                    messages_fn=lambda o: training_messages(o, self.model_id, self.hint))
        states = select_states(self.graph, rows)
        metrics = self.pool.refill(states)
        summ = summarize(episodes)
        metrics.update({f"onpolicy/{k}": v for k, v in summ.items() if k.startswith(("success", "invalid_rate", "path_len"))})
        cache: dict[int, dict[int, int]] = {}
        dist = [cache.setdefault(r["target"], self.graph.dist_to(r["target"]))[r["path"][-1]] for r in states]
        metrics.update({"onpolicy/states": len(states), "onpolicy/positions": len(rows),
                        "onpolicy/mean_state_dist": sum(dist) / len(dist) if dist else 0.0,
                        "onpolicy/seconds": time.time() - t0})
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                for e in episodes:
                    f.write(json.dumps({"round_step": step, **e.to_dict()}) + "\n")
        print(f"[onpolicy] step {step}: {len(games)} games, success {summ['success']:.2f}, "
              f"{len(states)} states kept of {len(rows)} positions, {time.time() - t0:.0f}s", flush=True)
        return metrics
