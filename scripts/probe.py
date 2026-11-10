"""Hands-on probe: several model configs answer the same real mid-game states, several samples each.

Example (Mac, about 10-25 s per answer):
  uv run python scripts/probe.py --episodes data/runs/<eval>.jsonl \
      --config stage2=data/ckpt/grpo-step-r1-stage2:0.6 --config stage3=data/ckpt/grpo-step-r1-stage3:0.6
Each --config is name=adapter_dir:temperature. Every answer goes to a crash-safe JSONL with the full model text.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch

from wikigame import WikiGameEnv, WikiGraph
from wikigame.llm import HFPolicy, build_messages
from wikigame.probe import collect_states, sample_states, score_answer

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True, help="eval JSONL whose visited states are replayed")
    ap.add_argument("--config", action="append", required=True, help="name=adapter_dir:temperature (repeatable)")
    ap.add_argument("--graph", default="data/wikispeedia.json")
    ap.add_argument("--dists", type=int, nargs="+", default=[2, 3, 4], help="distances to probe; the largest includes farther")
    ap.add_argument("--per-dist", type=int, default=4)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    graph = WikiGraph.load(a.graph)
    episodes = [json.loads(line) for line in Path(a.episodes).read_text().splitlines() if line]
    states = sample_states(collect_states(graph, episodes), a.dists, a.per_dist, a.seed)
    out = Path(a.out or f"data/runs/probe-{int(time.time())}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    env = WikiGameEnv(graph, seed=0)
    total = len(states) * a.samples * len(a.config)
    print(f"{len(states)} states, {len(a.config)} configs, {a.samples} samples each = {total} answers -> {out}", flush=True)

    rows, done, t0 = [], 0, time.time()
    for spec in a.config:
        name, rest = spec.split("=", 1)
        adapter, temperature = rest.rsplit(":", 1)
        policy = HFPolicy(MODEL, max_new_tokens=a.max_new_tokens, temperature=float(temperature), lenient=True, seed=a.seed,
                          adapter=adapter)
        for sid, state in enumerate(states):
            ids = [graph.id_of(t) for t in state["path"]]
            obs = env.set_state(ids[0], graph.id_of(state["target"]), ids)
            for sample in range(a.samples):
                text = policy.generate(build_messages(obs))
                row = {"config": name, "state": sid, "sample": sample, "dist": state["dist"], "target": state["target"],
                       "path": state["path"], "temperature": float(temperature), **score_answer(graph, state, text), "text": text}
                rows.append(row)
                done += 1
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{done}/{total}] {name} state {sid} d={state['dist']} -> {row['outcome']} ({row['chosen']}) "
                      f"{time.time() - t0:.0f}s", flush=True)
        del policy
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    d = pd.DataFrame(rows)
    d["dist"] = d.dist.clip(upper=max(a.dists))
    print(pd.crosstab([d.config, d.dist], d.outcome, normalize="index").round(2).to_string())
    print(f"answers in {out}")


if __name__ == "__main__":
    main()
