"""Where training groups come from: prompts in, G scored answers per prompt out.

``StepSource`` is the per-step setup every stage so far used: one prompt = one game state, its G answers are scored
by replaying that one move in the env (``grpo_reward.StepReward``, unchanged). ``PlayoutSource`` (v2) scores
each answer by playing its game out to the end with the current model instead. States come from a ``StatePool``
(scripted states, plus the latest on-policy round when ``OnPolicyRounds`` is on). A whole-game source (v2) will plug
in here with the same ``next_batch`` contract, without touching the algorithms or the learner.
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter
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


@dataclass
class PlayoutReturn:
    """How a played-out game is scored (v2, the user's choice: win + speed, partial credit for losses).

      win   1 - move_cost x moves used from the branch position on (the branching move included); with the
            10-move cap a win never scores below 0.5
      loss  loss_credit x share of the branch position's distance to the target closed by where the game ended
            (0 if it ended no closer), minus invalid_end_penalty if it ended on too many invalid answers
    Every win outscores every loss (0.5 > 0.3) when move_bonus is 0.

      move_bonus (stage 5c, default 0 = stages 5/5b): + this if the branching answer itself moved closer by graph
            distance, - this if it moved farther, 0 if sideways or invalid (an invalid answer already costs a move in
            its play-out). A per-move signal that does not depend on one noisy play-out; with move_bonus >= 0.1 a
            good move whose play-out lost can outrank a bad move whose play-out won."""
    move_cost: float = 0.05
    loss_credit: float = 0.3
    invalid_end_penalty: float = 0.3
    move_bonus: float = 0.0

    def bonus(self, moved: str) -> float:
        return {"closer": self.move_bonus, "farther": -self.move_bonus}.get(moved, 0.0)

    def score(self, success: bool, moves: int, d0: int | None, d_final: int | None, end_reason: str) -> float:
        if success:
            return 1.0 - self.move_cost * moves
        closed = max(0.0, (d0 - d_final) / d0) if d0 and d_final is not None else 0.0
        return self.loss_credit * closed - (self.invalid_end_penalty if end_reason == "invalid_action" else 0.0)


class PlayoutSource:
    """v2 'branch and play out': G answers to one game position, each played out to the end of the game by the
    current model, each answer scored by how its game ended (``PlayoutReturn``). Only the branching answers are
    trained; the play-out moves are sampled from the same policy (same temperature) but get no gradient.

    All G x prompts games advance together, one batched generate call per move (like ``parallel.play_games``)."""

    def __init__(self, pool: StatePool, graph: WikiGraph, tok, generator: Generator, *, num_generations: int,
                 max_tokens: int, temperature: float, model_id: str, hint: str | None, max_steps_game: int = 10,
                 feedback: bool = False, lenient: bool = False, returns: PlayoutReturn | None = None, seed: int = 0):
        self.pool, self.graph, self.tok, self.gen = pool, graph, tok, generator
        self.G, self.max_tokens, self.temperature = num_generations, max_tokens, temperature
        self.model_id, self.hint, self.max_steps_game = model_id, hint, max_steps_game
        self.feedback, self.lenient = feedback, lenient
        self.returns = returns or PlayoutReturn()
        self.rng = random.Random(seed)
        self.cursor = 0
        self._dist: dict[int, dict[int, int]] = {}

    def dist(self, target: int) -> dict[int, int]:
        if target not in self._dist:
            if len(self._dist) > 512:
                self._dist.clear()
            self._dist[target] = self.graph.dist_to(target)
        return self._dist[target]

    def _step(self, game: dict, text: str) -> None:
        from ..env import parse_action
        from ..llm import lenient_answer

        text = lenient_answer(text) if self.lenient else text
        game["obs"], _, term, trunc, info = game["env"].step(parse_action(text))
        game["moves"] += 1
        game["infos"].append(info)
        if term or trunc:
            game["done"] = True
            game["success"] = bool(info.get("success"))
            game["end"] = "success" if game["success"] else info.get("reason", "step_limit")

    def next_batch(self, n_prompts: int) -> tuple[list[Group], dict[str, float]]:
        from ..env import WikiGameEnv

        rows = [self.pool[self.cursor + i] for i in range(n_prompts)]
        self.cursor += n_prompts
        enc = [encode_prompt(self.tok, r["prompt"]) for r in rows]
        t0 = time.time()
        out = self.gen.generate([ids for ids, _ in enc], self.G, self.max_tokens, self.temperature)
        gen_s = time.time() - t0
        games = []
        for gi, (row, (_, opened), samples) in enumerate(zip(rows, enc, out)):
            for s in samples:
                env = WikiGameEnv(self.graph, max_steps=self.max_steps_game, feedback=self.feedback,
                                  seed=self.rng.randrange(2**31))
                env.set_state(row["source"], row["target"], row["path"], n_invalid=row["n_invalid"])
                game = {"group": gi, "env": env, "moves": 0, "infos": [], "done": False, "success": False,
                        "end": "", "text": restore_think(s.text, opened),
                        "d0": self.dist(row["target"]).get(row["path"][-1])}
                self._step(game, game["text"])  # the branching move: the answer being trained
                game["d1"] = self.dist(row["target"]).get(env.cur) if game["infos"][0]["valid"] else None
                games.append(game)
        branch_infos = [g["infos"][0] for g in games]
        generate = chat_batch_generate(self.gen, self.tok, self.max_tokens, self.temperature)
        t1, calls = time.time(), 0
        running = [g for g in games if not g["done"]]
        while running:  # play every unfinished game one move further per batched call
            texts = generate([training_messages(g["obs"], self.model_id, self.hint) for g in running])
            calls += 1
            for g, text in zip(running, texts):
                self._step(g, text)
            running = [g for g in running if not g["done"]]
        play_s = time.time() - t1
        groups, rets, parts = [], [], []
        for gi, (row, (ids, _), samples) in enumerate(zip(rows, enc, out)):
            mine = [g for g in games if g["group"] == gi]
            rs = []
            for g in mine:
                env = g["env"]
                d_final = self.dist(row["target"]).get(env.cur)
                ret = self.returns.score(g["success"], g["moves"], g["d0"], d_final, g["end"])
                bonus = self.returns.bonus(_moved(g))
                rs.append(ret + bonus)
                parts.append((ret, bonus))
            groups.append(Group(row, ids, samples, [g["text"] for g in mine], rs))
            rets += rs
        n = len(games)
        ends = Counter(g["end"] for g in games)
        metrics = {"time/generate_s": gen_s, "time/playout_s": play_s, "playout/calls": calls,
                   "playout/success": sum(g["success"] for g in games) / n,
                   "playout/moves_mean": sum(g["moves"] for g in games) / n,
                   **{f"playout/end_{k}": ends.get(k, 0) / n for k in ("success", "invalid_action", "step_limit", "dead_end")},
                   "env/invalid_rate": sum(not i["valid"] for i in branch_infos) / n,
                   "env/closer_rate": sum(_moved(g) == "closer" for g in games) / n,
                   "env/sideways_rate": sum(_moved(g) == "sideways" for g in games) / n,
                   "playout/return_mean": sum(r for r, _ in parts) / n,  # the play-out score alone (= reward in 5/5b)
                   "playout/bonus_mean": sum(b for _, b in parts) / n,
                   "onpolicy/drawn_this_step": sum(r["kind"] == "onpolicy" for r in rows) / len(rows)}
        for cause, k in Counter(i["cause"] for i in branch_infos if i.get("cause")).items():
            metrics[f"env/invalid/{cause}"] = k / n
        for label, sel in (("d1", lambda d: d == 1), ("d2", lambda d: d == 2), ("d3", lambda d: d == 3),
                           ("d4p", lambda d: d is not None and d >= 4)):  # same keys as the TRL step-reward runs
            mine = [g for g in games if sel(g["d0"])]
            metrics[f"env/{label}/share"] = len(mine) / n
            if mine:
                metrics[f"env/{label}/closer_rate"] = sum(_moved(g) == "closer" for g in mine) / len(mine)
                metrics[f"env/{label}/invalid_rate"] = sum(not g["infos"][0]["valid"] for g in mine) / len(mine)
        win = [g["moves"] for g in games if g["success"]]
        if win:
            metrics["playout/moves_to_win"] = sum(win) / len(win)
        return groups, metrics


def _moved(game: dict) -> str:
    """What the branching move did, from graph distances (the playout envs have no shaping, so no `progress`)."""
    if not game["infos"][0]["valid"]:
        return "invalid"
    d0, d1 = game["d0"], game["d1"]
    if d0 is None or d1 is None or d1 > d0:
        return "farther"
    return "closer" if d1 < d0 else "sideways"


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
