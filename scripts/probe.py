"""Hands-on probe: several model configs answer the same real mid-game states, several samples each.

Example (Mac, about 10-25 s per answer):
  uv run python scripts/probe.py --episodes data/runs/<eval>.jsonl \
      --config stage2=data/ckpt/grpo-step-r1-stage2:0.6 --config stage3=data/ckpt/grpo-step-r1-stage3:0.6
Each --config is name=adapter_dir:temperature. Every answer goes to a crash-safe JSONL with the full model text.

Prompt-hint comparison (stage-4 prep): states come from the TRAINING sampler (held-out eval pairs are never used, so
the wording is not picked on the eval set) and every hint answers every state:
  uv run python scripts/probe.py --train-states 34 --hints brief bridge compare plan \
      --config stage3=data/ckpt/grpo-step-r1-stage3:0.6 --samples 1
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
from wikigame.llm import HFPolicy, build_messages, resolve_hint
from wikigame.probe import collect_states, sample_states, score_answer, training_states

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--episodes", help="eval JSONL whose visited states are replayed")
    src.add_argument("--train-states", type=int, metavar="N",
                     help="instead: N states per distance from the training sampler (no held-out pairs)")
    ap.add_argument("--hints", nargs="+", default=[None],
                    help="prompt hints to compare on the same states: names from wikigame.llm.HINTS or literal text")
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
    if a.train_states:
        states = training_states(graph, a.dists, a.train_states, a.seed)
    else:
        episodes = [json.loads(line) for line in Path(a.episodes).read_text().splitlines() if line]
        states = sample_states(collect_states(graph, episodes), a.dists, a.per_dist, a.seed)
    out = Path(a.out or f"data/runs/probe-{int(time.time())}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    env = WikiGameEnv(graph, seed=0)
    total = len(states) * a.samples * len(a.config) * len(a.hints)
    print(f"{len(states)} states, {len(a.config)} configs, {len(a.hints)} hints, {a.samples} samples each = {total} answers -> {out}", flush=True)

    rows, done, t0 = [], 0, time.time()
    for spec in a.config:
        name, rest = spec.split("=", 1)
        adapter, temperature = rest.rsplit(":", 1)
        policy = HFPolicy(MODEL, max_new_tokens=a.max_new_tokens, temperature=float(temperature), lenient=True, seed=a.seed,
                          adapter=adapter)
        default_hint = policy.hint
        for sid, state in enumerate(states):
            ids = [graph.id_of(t) for t in state["path"]]
            obs = env.set_state(ids[0], graph.id_of(state["target"]), ids)
            for hint in a.hints:  # hints are the inner loop, so a stopped run still has every hint on the same states
                policy.hint = default_hint if hint is None else (resolve_hint(hint) or None)
                label = name if hint is None else f"{name}/{hint if len(hint) < 20 else 'custom'}"
                for sample in range(a.samples):
                    before = policy.new_tokens
                    text = policy.generate(build_messages(obs))
                    row = {"config": label, "state": sid, "sample": sample, "dist": state["dist"], "target": state["target"],
                           "path": state["path"], "temperature": float(temperature), "hint": hint,
                           "new_tokens": policy.new_tokens - before, **score_answer(graph, state, text), "text": text}
                    rows.append(row)
                    done += 1
                    with out.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"[{done}/{total}] {label} state {sid} d={state['dist']} -> {row['outcome']} ({row['chosen']}) "
                          f"{row['new_tokens']} tok {time.time() - t0:.0f}s", flush=True)
        del policy
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    d = pd.DataFrame(rows)
    d["dist"] = d.dist.clip(upper=max(a.dists))
    print(pd.crosstab([d.config, d.dist], d.outcome, normalize="index").round(2).to_string())
    print(d.groupby("config").new_tokens.agg(["mean", "max"]).round(0).to_string())
    print(f"answers in {out}")


if __name__ == "__main__":
    main()
