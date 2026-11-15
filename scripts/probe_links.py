"""Knowledge probe: does the model know which pages link to the target?

At positions 2 clicks from the target (from an eval's episodes) the right move is a link whose page links to the target.
The model cannot see that page, so choosing well needs recalled knowledge. For each position, ask about up to
--per-side links that DO link to the target (graph distance 1) and as many that do NOT (distance >= 2):
  'Does the Wikipedia article "X" contain a link to the article "TARGET"?'  -> <answer>yes|no</answer>
and compare yes-rates. Caveat: the graph is the 2007 Wikispeedia subset; the model knows (some of) today's Wikipedia.

Pod (vLLM):
  uv run python scripts/probe_links.py --episodes data/runs/<eval>.jsonl --adapters none data/ckpt/grpo-step-r1-stage4 \
      --out data/runs/probe-links.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import pandas as pd

from wikigame import WikiGraph
from wikigame.probe import collect_states, sample_states

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
QUESTION = ('Question about Wikipedia. Does the Wikipedia article "{x}" contain a hyperlink to the article "{y}"? '
            "Think briefly (a few sentences), then answer with <answer>yes</answer> or <answer>no</answer>.")
ANSWER = re.compile(r"<answer>\s*(yes|no)\s*</answer>", re.I)
BARE = re.compile(r"\b(yes|no)\b", re.I)


def parse_yes_no(text: str) -> str | None:
    """<answer>yes|no</answer>, else the first bare yes/no after the reasoning (</think>). None if cut off."""
    m = ANSWER.search(text)
    if m:
        return m.group(1).lower()
    if "</think>" not in text:
        return None
    m = BARE.search(text.split("</think>", 1)[1])
    return m.group(1).lower() if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--adapters", nargs="+", default=["none"], help="adapter dirs; 'none' = the base model")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--graph", default="data/wikispeedia.json")
    ap.add_argument("--positions", type=int, default=150)
    ap.add_argument("--per-side", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--vllm-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer, GenerationConfig

    from wikigame.rl.generation import VLLMGenerator, encode_prompt, restore_think

    graph = WikiGraph.load(a.graph)
    episodes = [json.loads(line) for line in Path(a.episodes).read_text().splitlines() if line]
    # exactly 2 clicks (sample_states' largest distance means "that far or farther")
    states = sample_states([s for s in collect_states(graph, episodes) if s["dist"] == 2], [2], a.positions, a.seed)
    rng = random.Random(a.seed)
    queries = []
    for sid, s in enumerate(states):
        ids = [graph.id_of(t) for t in s["path"]]
        target = graph.id_of(s["target"])
        dist = graph.dist_to(target)
        links = [n for n in graph.adj[ids[-1]] if n not in set(ids)]
        pos = [n for n in links if dist.get(n) == 1]
        neg = [n for n in links if dist.get(n, 99) >= 2]
        for label, pool in (("links", pos), ("no_link", neg)):
            for n in rng.sample(pool, min(a.per_side, len(pool))):
                queries.append({"state": sid, "here": s["path"][-1], "target": s["target"], "page": graph.titles[n],
                                "truth": label, "n_links_on_page": len(links), "n_linking_options": len(pos)})
    print(f"{len(states)} positions, {len(queries)} questions "
          f"({sum(q['truth'] == 'links' for q in queries)} true links)", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    gcfg = GenerationConfig.from_pretrained(a.model)
    top_p = gcfg.top_p if (gcfg.do_sample and gcfg.top_p) else 1.0
    first = None if a.adapters[0] == "none" else a.adapters[0]
    gen = VLLMGenerator(a.model, gpu_memory_utilization=a.vllm_mem, seed=a.seed, adapter=first, top_p=top_p)
    enc = [encode_prompt(tok, [{"role": "user", "content": QUESTION.format(x=q["page"], y=q["target"])}]) for q in queries]
    rows = []
    for adapter in a.adapters:
        gen.set_adapter(None if adapter == "none" else adapter)
        t0 = time.time()
        out = gen.generate([ids for ids, _ in enc], 1, a.max_new_tokens, a.temperature)
        print(f"{adapter}: {len(queries)} answers in {time.time() - t0:.0f}s", flush=True)
        for q, o, (_, opened) in zip(queries, out, enc):
            text = restore_think(o[0].text, opened)
            rows.append({**q, "model": "base" if adapter == "none" else Path(adapter).name,
                         "answer": parse_yes_no(text), "text": text})
    path = Path(a.out or f"data/runs/probe-links-{int(time.time())}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    d = pd.DataFrame(rows)
    d["yes"] = d.answer == "yes"
    d["unparsed"] = d.answer.isna()
    t = d.groupby(["model", "truth"]).agg(n=("yes", "size"), yes_rate=("yes", "mean"), unparsed=("unparsed", "mean"))
    print((t.assign(yes_rate=t.yes_rate * 100, unparsed=t.unparsed * 100)).round(1).to_string())
    for m, g in d.groupby("model"):
        tpr = g[g.truth == "links"].yes.mean()
        fpr = g[g.truth == "no_link"].yes.mean()
        print(f"{m}: says yes to {tpr:.1%} of real links and {fpr:.1%} of non-links; balanced accuracy {(tpr + 1 - fpr) / 2:.1%}")
    print(f"rows in {path}")


if __name__ == "__main__":
    main()
