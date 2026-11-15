"""Step 1 of the value plan: play the held-out games with value-ranked moves (no generation, no training).

At every move each visible link is scored by the value model ("how many clicks from this link to the target?",
expected distance over the digits 1-5) and the lowest is clicked; a linked target is clicked directly. Games, envs
and seeds are the standard eval's, so the episodes compare game by game with every model eval (Statistics page).

Pod:
  uv run python scripts/value_play.py --adapter data/ckpt/value-step0/best --name value-ranked-step0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from wikigame import WikiGraph
from wikigame.eval import make_pairs, summarize
from wikigame.tracking import detect_machine, init_run, log_episodes, log_eval
from wikigame.value import ValueScorer, play_value_games

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="value-trained LoRA adapter (value_train.py's best/ or final/)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--graph", default="data/wikispeedia.json")
    ap.add_argument("--graph-name", default="wikispeedia")
    ap.add_argument("--hops", type=int, nargs="+", default=[2, 3, 4], help="2+ hops only, the project's focus")
    ap.add_argument("--n-per-hop", type=int, default=50)
    ap.add_argument("--pair-pool", type=int, default=50, help="the shared eval pair set to take the first n from")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--batch-games", type=int, default=64)
    ap.add_argument("--score-batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    graph = WikiGraph.load(a.graph)
    pool = make_pairs(graph, hops=tuple(a.hops), n_per_hop=a.pair_pool)
    games = [(s, t, k) for k, plist in pool.items() for s, t in plist[: a.n_per_hop]]
    t0 = time.time()
    scorer = ValueScorer(graph, a.model, a.adapter, device=a.device, batch=a.score_batch)
    print(f"loaded {a.model} + {a.adapter} on {scorer.device} in {time.time() - t0:.0f}s, {len(games)} games", flush=True)

    name = a.name or f"value-ranked-{Path(a.adapter).parent.name}-{Path(a.adapter).name}"
    out = Path("data/runs")
    out.mkdir(parents=True, exist_ok=True)
    local = out / f"{name}-{detect_machine()}-{int(t0)}.jsonl"
    done = []

    def finished(i, ep) -> None:
        done.append(ep)
        with local.open("a") as f:
            f.write(json.dumps(ep.to_dict()) + "\n")
        print(f"[{len(done)}/{len(games)}] hop{ep.n_hops} {ep.source} -> {ep.target}: {ep.end_reason} "
              f"(moves {ep.n_valid}) | {scorer.questions} questions scored, {time.time() - t0:.0f}s", flush=True)

    episodes = play_value_games(graph, scorer, games, max_steps=a.max_steps, batch_size=a.batch_games, seed=a.seed,
                                on_episode=finished)
    metrics = summarize(episodes)
    metrics.update({"value_questions": scorer.questions, "seconds": time.time() - t0})
    print("\n" + " ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k.startswith(("success", "invalid", "path_len"))))
    if not a.no_wandb:
        cfg = {"policy": name, "model": a.model, "adapter": a.adapter, "decoding": "value-ranked (greedy argmin)",
               "hops": a.hops, "n_per_hop": a.n_per_hop, "max_steps": a.max_steps, "seed": a.seed,
               "adapter_merged": False}
        run = init_run(name, "eval", cfg, graph, a.graph_name, tags=["value-play", "step1", "backend:hf-value"])
        log_eval(run, metrics, step=0)
        log_episodes(run, episodes, step=0, name="eval")
        print("W&B:", run.url)
        run.finish()
    print("local copy:", local)


if __name__ == "__main__":
    main()
