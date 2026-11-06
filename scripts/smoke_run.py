"""Baseline smoke run: evaluate the random floor and oracle ceiling and log both to W&B.

Exercises the whole tracking pipeline (run identity, config, eval metrics, episode table + artifact)
without any model, on whichever machine you run it from.

  uv run python scripts/smoke_run.py                      # logs per WANDB_MODE (online after `wandb login`)
  WANDB_MODE=offline uv run python scripts/smoke_run.py   # no network; sync later with `wandb sync`
"""
from __future__ import annotations

import argparse

from wikigame import WikiGameEnv, WikiGraph
from wikigame.eval import evaluate, make_pairs
from wikigame.rollout import oracle_policy, random_policy
from wikigame.tracking import init_run, log_episodes, log_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/wikispeedia.json")
    ap.add_argument("--graph-name", default="wikispeedia")
    ap.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--n-per-hop", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    graph = WikiGraph.load(a.graph)
    pairs = make_pairs(graph, hops=tuple(a.hops), n_per_hop=a.n_per_hop)
    policies = {"random": random_policy(a.seed), "oracle": oracle_policy()}

    for name, policy in policies.items():
        env = WikiGameEnv(graph, max_steps=a.max_steps, seed=a.seed)
        cfg = {"policy": name, "hops": a.hops, "n_per_hop": a.n_per_hop, "max_steps": a.max_steps,
               "seed": a.seed, "reward": vars(env.rc), "max_links_shown": env.max_links_shown}
        run = init_run(f"baseline-{name}", "baseline", cfg, graph, a.graph_name, tags=["smoke"])
        metrics, episodes = evaluate(env, policy, pairs)
        log_eval(run, metrics, step=0)
        log_episodes(run, episodes, step=0, name="eval")
        print(f"{name:7s} " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k.startswith("success")))
        run.finish()


if __name__ == "__main__":
    main()
