"""Value step 2, data: positions toward training targets, the policy's own sampled reasoning, the value's top link.

Writes one JSONL row per position (train rows toward training targets, val rows toward validation targets):
prompt messages, the kept reasoning, the label (value's top link), whether a sample already chose it, and true graph
distances for analysis (labels never use them). Pod (vLLM + HF value scorer on the same card):
  uv run python scripts/distill_data.py --adapter data/ckpt/value-step0/best --out data/distill/step2.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from wikigame import WikiGraph
from wikigame.distill import choose_reasoning, label_for, sample_answer, strict_states
from wikigame.value import ValueScorer, make_splits

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="the step-0 value adapter: samples reasoning AND labels (frozen)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--graph", default="data/wikispeedia.json")
    ap.add_argument("--positions", type=int, default=3000)
    ap.add_argument("--val-positions", type=int, default=150)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7, help="the RL stages' training temperature")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    ap.add_argument("--score-batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/distill/step2.jsonl")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    from wikigame.rl.generation import VLLMGenerator, encode_prompt

    graph = WikiGraph.load(a.graph)
    splits = make_splits(graph, seed=0)  # the same splits as step 0
    states = [("train", s) for s in strict_states(graph, a.positions, a.model, splits, set(splits.train_targets), a.seed)]
    states += [("val", s) for s in strict_states(graph, a.val_positions, a.model, splits, set(splits.val_targets), a.seed + 1)]
    print(f"{len(states)} positions ({sum(k == 'train' for k, _ in states)} train, "
          f"{sum(k == 'val' for k, _ in states)} val)", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    gen = VLLMGenerator(a.model, gpu_memory_utilization=a.vllm_mem, seed=a.seed, adapter=a.adapter)
    enc = [encode_prompt(tok, s["prompt"]) for _, s in states]
    outs = gen.generate([ids for ids, _ in enc], a.samples, a.max_tokens, a.temperature)
    print(f"sampled {a.samples} answers x {len(states)} positions in {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    scorer = ValueScorer(graph, a.model, a.adapter, batch=a.score_batch)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats, n_rows = Counter(), 0
    with out.open("w") as f:
        for (split, s), samples, (ids, _) in zip(states, outs, enc):
            t = s["target"]
            links = s["links"]
            tlinked = graph.titles[t] in links
            scores = [] if tlinked else scorer([(graph.id_of(x), t) for x in links])
            label = label_for(graph, t, links, scores, rng)
            raw = [x.text for x in samples]
            picked = choose_reasoning(raw, label)
            d = graph.dist_to(t)
            here = s["path"][-1]
            answers = [sample_answer(x) for x in raw]
            stats["positions"] += 1
            if picked is None:
                stats["dropped_all_cut_off"] += 1
                continue
            reasoning, matched = picked
            dl = d.get(graph.id_of(label))
            row = {"split": split, "prompt": s["prompt"], "prompt_ids": ids, "reasoning": reasoning, "label": label,
                   "matched": matched, "target": t, "path": s["path"], "kind": s["kind"], "dist": d.get(here),
                   "label_dist": dl, "label_closer": dl is not None and dl < d.get(here, 99),
                   "sample_answers": answers, "target_linked": tlinked}
            f.write(json.dumps(row) + "\n")
            n_rows += 1
            stats["matched"] += matched
            stats["label_closer"] += row["label_closer"]
            ok = [x for x in answers if x is not None and graph.id_of(x) is not None]
            stats["samples_closer"] += sum(d.get(graph.id_of(x), 99) < d.get(here, 99) for x in ok)
            stats["samples"] += len(answers)
    print(f"value scoring + rows in {time.time() - t1:.0f}s ({scorer.questions} value questions)", flush=True)
    print(f"rows {n_rows} of {stats['positions']} (dropped, all samples cut off: {stats['dropped_all_cut_off']}) | "
          f"reasoning already chose the label: {stats['matched'] / max(1, n_rows):.1%} | label moves closer (true graph): "
          f"{stats['label_closer'] / max(1, n_rows):.1%} | policy samples move closer: "
          f"{stats['samples_closer'] / max(1, stats['samples']):.1%} -> {out}", flush=True)


if __name__ == "__main__":
    main()
