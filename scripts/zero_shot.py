"""Zero-shot baseline for a prompted LLM on the eval pairs; logs to W&B like the random/oracle baselines.

Uses the first N pairs of each hop from the SAME held-out pair set as scripts/smoke_run.py (50 per hop,
seed 1234), so results are directly comparable with the random/oracle baselines.

  HF_HOME=data/hf uv run python scripts/zero_shot.py --n-per-hop 10        # Mac (mps), small
  HF_HOME=data/hf uv run python scripts/zero_shot.py --n-per-hop 50        # CUDA box, full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from wikigame import WikiGameEnv, WikiGraph
from wikigame.eval import make_pairs, summarize
from wikigame.llm import DEFAULT_MODEL, HFPolicy
from wikigame.rollout import run_episode
from wikigame.tracking import detect_machine, init_run, log_episodes, log_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--graph", default="data/wikispeedia.json")
    ap.add_argument("--graph-name", default="wikispeedia")
    ap.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--n-per-hop", type=int, default=10)
    ap.add_argument("--pair-pool", type=int, default=50, help="size of the shared eval pair set to take the first n from")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default); R1 distills want ~0.6")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (only used when temperature > 0)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--lenient", action="store_true",
                    help="also accept a bare/```json title (separates format failures from navigation failures)")
    ap.add_argument("--feedback", action="store_true",
                    help="env tells the model why its last answer was invalid (changes the observation)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir from GRPO training (evaluates the trained model)")
    ap.add_argument("--name", default=None, help="policy/experiment name for W&B (default derived from model+flags)")
    ap.add_argument("--no-wandb", action="store_true", help="skip W&B (local JSONL only)")
    a = ap.parse_args()

    graph = WikiGraph.load(a.graph)
    pool = make_pairs(graph, hops=tuple(a.hops), n_per_hop=a.pair_pool)
    pairs = {k: v[: a.n_per_hop] for k, v in pool.items()}
    t0 = time.time()
    policy = HFPolicy(a.model, device=a.device, max_new_tokens=a.max_new_tokens, lenient=a.lenient,
                      temperature=a.temperature, seed=a.seed, adapter=a.adapter)
    print(f"loaded {a.model} on {policy.device} in {time.time() - t0:.0f}s", flush=True)

    short = a.model.split("/")[-1].lower()
    name = a.name or (f"zeroshot-{short}" + ("-lenient" if a.lenient else "") + ("-feedback" if a.feedback else ""))
    cfg = {"policy": name, "model": a.model, "decoding": "greedy" if a.temperature == 0 else f"sample(T={a.temperature})",
           "temperature": a.temperature, "sample_seed": a.seed, "system_in_user": policy.system_in_user, "prompt_hint": policy.hint, "adapter": a.adapter, "max_new_tokens": a.max_new_tokens, "hops": a.hops,
           "n_per_hop": a.n_per_hop, "max_steps": a.max_steps, "seed": 0, "device": policy.device,
           "lenient_parse": a.lenient, "feedback": a.feedback}
    env = WikiGameEnv(graph, max_steps=a.max_steps, seed=0, feedback=a.feedback)
    cfg.update(reward=vars(env.rc), max_links_shown=env.max_links_shown)

    out = Path("data/runs"); out.mkdir(parents=True, exist_ok=True)
    local = out / f"{name}-{detect_machine()}-{int(t0)}.jsonl"
    episodes, total = [], sum(len(v) for v in pairs.values())
    for k, plist in pairs.items():
        for s, t in plist:
            ep = run_episode(env, policy, options={"source": s, "target": t}, n_hops=k)
            episodes.append(ep)
            with local.open("a") as f:  # crash-safe: every finished episode is on disk immediately
                f.write(json.dumps(ep.to_dict()) + "\n")
            st = policy.stats()
            print(f"[{len(episodes)}/{total}] hop{k} {ep.source} -> {ep.target}: {ep.end_reason} "
                  f"(valid {ep.n_valid}, invalid {ep.n_invalid}) | {st['sec_per_call']:.1f}s/call, "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)

    metrics = summarize(episodes)
    metrics.update({f"gen_{k}": v for k, v in policy.stats().items()})
    print("\n" + " ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k.startswith(("success", "invalid", "path_len"))))
    if not a.no_wandb:
        run = init_run(name, "eval" if a.adapter else "baseline", cfg, graph, a.graph_name,
                       tags=["grpo-eval" if a.adapter else "zero-shot", "llm"])
        log_eval(run, metrics, step=0)
        log_episodes(run, episodes, step=0, name="eval")
        print("W&B:", run.url)
        run.finish()
    print("local copy:", local)


if __name__ == "__main__":
    main()
