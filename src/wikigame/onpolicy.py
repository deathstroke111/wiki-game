"""On-policy state rounds for per-step GRPO (stage-4 V2, option 1).

Every ``every`` optimizer steps (and once before step 1) the model being trained plays a round of whole games in
parallel (``parallel.play_games``) on fresh training pairs. The positions it reached become training states, mixed
with the scripted ones (``states.sample_states``); scoring is unchanged (``grpo_reward.StepReward`` replays one move).
This closes the gap between the scripted states and the positions the model's own play leads to, e.g. six moves in,
two invalid answers used, parked on an odd page.

How the mix works without touching TRL's sampler: TRL draws a shuffled list of row indices up front and fetches each
row when it is needed (G times in a row, one per sampled answer). ``StatePool`` decides a row's content the first
time its index is fetched, from the latest round's states with probability ``share`` (while any are left), otherwise
from the next scripted state, and then keeps it, so all G answers of a group see the same state. States from older
rounds are dropped when a new round arrives, so the on-policy states are never more than one round stale.
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

from .eval import heldout_pairs, summarize
from .graph import WikiGraph
from .llm import batch_generate, training_messages
from .parallel import play_games


class StatePool:
    """Map-style dataset for GRPOTrainer: scripted rows plus the latest on-policy round, mixed at ``share``."""

    def __init__(self, scripted: list[dict], share: float, seed: int = 0):
        if not scripted:
            raise ValueError("StatePool needs at least one scripted row")
        self.scripted, self.share = scripted, share
        # TRL builds the reward kwargs from the first row's keys and indexes every row with them, so all rows must
        # share one column set: on-policy rows are cut down to the scripted columns (play_games adds e.g. "game")
        self.columns = tuple(scripted[0])
        self.rng = random.Random(seed)
        self.fresh: list[dict] = []
        self.assigned: dict[int, dict] = {}
        self.next_scripted = 0
        self.drawn = Counter()  # draws per source since the last refill: 'onpolicy' / 'scripted'

    def __len__(self) -> int:
        return len(self.scripted)

    def __getitem__(self, i: int) -> dict:
        if i not in self.assigned:
            if self.fresh and self.rng.random() < self.share:
                row, src = self.fresh.pop(), "onpolicy"
            else:
                row, src = self.scripted[self.next_scripted % len(self.scripted)], "scripted"
                self.next_scripted += 1
            self.assigned[i] = row
            self.drawn[src] += 1
        return self.assigned[i]

    def refill(self, rows: list[dict]) -> dict[str, float]:
        """Replace the on-policy states with a new round's. Returns how the draws since the last refill were split."""
        n = sum(self.drawn.values())
        stats = {"onpolicy/drawn_share": self.drawn["onpolicy"] / n if n else 0.0,
                 "onpolicy/unused_from_last_round": len(self.fresh)}
        self.fresh = [{k: r[k] for k in self.columns} for r in rows]  # KeyError here if a column is missing
        self.rng.shuffle(self.fresh)
        self.drawn.clear()
        return stats


def select_states(graph: WikiGraph, rows: list[dict]) -> list[dict]:
    """Training states from a round's positions: one per distinct (source, target, path), keeping the first visit
    (fewest invalid answers used), and only positions from which the target can still be reached at all."""
    dists: dict[int, dict[int, int]] = {}
    seen, out = set(), []
    for r in rows:
        key = (r["source"], r["target"], tuple(r["path"]))
        if key in seen:
            continue
        seen.add(key)
        d = dists.setdefault(r["target"], graph.dist_to(r["target"]))
        if r["path"][-1] in d:
            out.append(r)
    return out


def round_pairs(graph: WikiGraph, n: int, hops: tuple[int, ...], rng: random.Random,
                held: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    games = []
    while len(games) < n:
        k = rng.choice(hops)
        s, t = graph.sample_pair(k, rng)
        if (s, t) not in held:
            games.append((s, t, k))
    return games


def make_callback(trainer, pool: StatePool, graph: WikiGraph, *, every: int, n_games: int, hops: tuple[int, ...],
                  model_id: str, hint: str | None, max_steps: int, max_new_tokens: int, temperature: float,
                  batch_size: int, feedback: bool, seed: int, log_path: Path | None = None):
    """A TrainerCallback that plays a round before step 1 and after every ``every`` optimizer steps."""
    import torch
    from transformers import TrainerCallback
    from trl.models import unwrap_model_for_generation

    held = heldout_pairs(graph)
    tok = trainer.processing_class

    def play_round(step: int) -> None:
        t0 = time.time()
        rng = random.Random(seed * 100_003 + step)
        games = round_pairs(graph, n_games, hops, rng, held)
        model = trainer.model
        was_training = model.training
        with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator,
                                         gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs) as m:
            m.eval()
            try:
                def generate(batch):
                    texts = [tok.apply_chat_template(b, tokenize=False, add_generation_prompt=True) for b in batch]
                    return batch_generate(m, tok, texts, max_new_tokens, temperature)[0]

                with torch.no_grad():
                    # lenient like the held-out evals, so rounds reach the positions eval-time play reaches
                    episodes, rows = play_games(graph, generate, games, batch_size=batch_size, max_steps=max_steps,
                                                feedback=feedback, lenient=True, seed=rng.randrange(2**31),
                                                messages_fn=lambda o: training_messages(o, model_id, hint))
            finally:
                model.train(was_training)
        states = select_states(graph, rows)
        metrics = pool.refill(states)
        summ = summarize(episodes)
        metrics.update({f"onpolicy/{k}": v for k, v in summ.items()
                        if k.startswith(("success", "invalid_rate", "path_len"))})
        cache: dict[int, dict[int, int]] = {}
        dist = [cache.setdefault(r["target"], graph.dist_to(r["target"]))[r["path"][-1]] for r in states]
        metrics.update({"onpolicy/states": len(states), "onpolicy/positions": len(rows),
                        "onpolicy/mean_state_dist": sum(dist) / len(dist) if dist else 0.0,
                        "onpolicy/seconds": time.time() - t0})
        for k, v in metrics.items():
            trainer._metrics["train"][k].append(float(v))  # TRL averages these into the next logged step
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as f:
                for e in episodes:
                    f.write(json.dumps({"round_step": step, **e.to_dict()}) + "\n")
        print(f"[onpolicy] step {step}: {len(games)} games, success {summ['success']:.2f}, "
              f"{len(states)} states kept of {len(rows)} positions, {time.time() - t0:.0f}s", flush=True)

    class OnPolicyRounds(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            play_round(state.global_step)

        def on_step_end(self, args, state, control, **kw):
            if state.global_step % every == 0 and state.global_step < state.max_steps:
                play_round(state.global_step)

    return OnPolicyRounds()
