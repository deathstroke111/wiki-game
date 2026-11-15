"""Top-2 probe: does the right link sit in the model's second guess?

On the same game positions (taken from an eval's episodes, at 2 / 3 / 4+ clicks from the target) compare
  single  the normal one-link prompt, sampled twice (what two answers of a training group look like today)
  top2    one answer to a prompt asking for the best link AND a different second-best link
and score every named link as closer / sideways / farther / invalid. The question is coverage: how often at least one
of the two links moves closer, and whether asking for two distinct guesses beats two independent samples.

Pod (vLLM, a few minutes):
  uv run python scripts/probe_top2.py --episodes data/runs/<stage5-final-eval>.jsonl \
      --adapter data/ckpt/rl-trl-dapo-playout-r1-1.5b-stage5 --out data/runs/probe-top2-stage5.jsonl
No W&B run: a diagnostic, rows go to a JSONL and a summary table is printed.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from wikigame import WikiGameEnv, WikiGraph
from wikigame.env import SYSTEM_PROMPT, classify_invalid, parse_action
from wikigame.llm import R1_HINT, lenient_answer, merge_system
from wikigame.probe import collect_states, sample_states

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TOP2_PROMPT = (
    "You are playing the Wikipedia Game. Starting from the current article, choose the TWO links most likely to reach "
    "the target article in as few steps as possible: your best link first, then a DIFFERENT second-best link. Think "
    'inside <think></think>, then answer with JSON inside <answer></answer>, e.g. '
    '<answer>{"titles": ["Best Link Title", "Second Best Link Title"]}</answer>. '
    "Both titles MUST be copied exactly from the list of links."
)
TOP2_HINT = ("Reason briefly (a few sentences). Do NOT go through the links one by one: name the two most promising "
             "links for reaching the target, then answer immediately with the <answer> tag.")
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def parse_top2(text: str) -> list[str] | None:
    """The titles list from <answer>{"titles": [...]}</answer>; falls back to every quoted string in the answer.
    None if there is no answer block at all."""
    m = ANSWER.search(text)
    if not m:
        return None
    body = m.group(1).strip()
    try:
        obj = json.loads(body)
        titles = obj.get("titles") if isinstance(obj, dict) else None
        if isinstance(titles, list) and all(isinstance(t, str) for t in titles):
            return titles
    except json.JSONDecodeError:
        pass
    quoted = re.findall(r'"([^"]+)"', body)
    return [q for q in quoted if q not in ("titles", "title")] or None


def outcome(graph: WikiGraph, ids: list[int], target: int, dist: dict[int, int], title: str | None) -> str:
    cause = classify_invalid(graph, ids[-1], set(ids), target, title)
    if cause is not None:
        return "invalid"
    before, after = dist.get(ids[-1]), dist.get(graph.id_of(title))
    if before is None or after is None or after > before:
        return "farther"
    return "closer" if after < before else "sideways"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True, help="eval JSONL whose visited positions are probed")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--graph", default="data/wikispeedia.json")
    ap.add_argument("--dists", type=int, nargs="+", default=[2, 3, 4], help="the largest includes farther positions")
    ap.add_argument("--per-dist", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--vllm-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--coverage", type=int, default=0, metavar="K",
                    help="instead: K samples of the normal prompt per position at each --temperatures value; reports "
                         "coverage@k (some sample moves closer), distinct links and invalid rate (no top-2 prompt)")
    ap.add_argument("--temperatures", type=float, nargs="+", default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer, GenerationConfig

    from wikigame.rl.generation import VLLMGenerator, encode_prompt, restore_think

    graph = WikiGraph.load(a.graph)
    episodes = [json.loads(line) for line in Path(a.episodes).read_text().splitlines() if line]
    states = sample_states(collect_states(graph, episodes), a.dists, a.per_dist, a.seed)
    env = WikiGameEnv(graph, seed=0)
    tok = AutoTokenizer.from_pretrained(a.model)
    gcfg = GenerationConfig.from_pretrained(a.model)
    top_p = gcfg.top_p if (gcfg.do_sample and gcfg.top_p) else 1.0
    gen = VLLMGenerator(a.model, gpu_memory_utilization=a.vllm_mem, seed=a.seed, adapter=a.adapter, top_p=top_p)

    single_msgs, top2_msgs, meta = [], [], []
    for sid, s in enumerate(states):
        ids = [graph.id_of(t) for t in s["path"]]
        target = graph.id_of(s["target"])
        obs = env.set_state(ids[0], target, ids)
        single_msgs.append(merge_system([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}], R1_HINT))
        top2_msgs.append(merge_system([{"role": "system", "content": TOP2_PROMPT}, {"role": "user", "content": obs}], TOP2_HINT))
        meta.append((sid, s, ids, target, graph.dist_to(target)))

    def run(msgs, n):
        enc = [encode_prompt(tok, m) for m in msgs]
        out = gen.generate([ids for ids, _ in enc], n, a.max_new_tokens, a.temperature)
        return [[restore_think(x.text, opened) for x in o] for o, (_, opened) in zip(out, enc)]

    if a.coverage:
        coverage(a, graph, meta, single_msgs, run)
        return
    t0 = time.time()
    singles = run(single_msgs, 2)
    top2s = run(top2_msgs, 1)
    print(f"{len(states)} positions, {3 * len(states)} answers in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for (sid, s, ids, target, dist), sing, (t2,) in zip(meta, singles, top2s):
        picks = [parse_action(lenient_answer(t)) for t in sing]
        titles = parse_top2(t2)
        g1, g2 = (titles + [None, None])[:2] if titles else (None, None)
        links = [n for n in graph.adj[ids[-1]] if n not in set(ids)]
        chance = sum(dist.get(n, 10**9) < dist[ids[-1]] for n in links) / len(links) if links else 0.0
        rows.append({
            "state": sid, "dist": s["dist"], "target": s["target"], "path": s["path"], "chance_closer": chance,
            "s1": picks[0], "s2": picks[1],
            "s1_out": outcome(graph, ids, target, dist, picks[0]), "s2_out": outcome(graph, ids, target, dist, picks[1]),
            "top2_titles": titles, "g1": g1, "g2": g2, "top2_format_ok": bool(titles) and len(titles) >= 2,
            "g1_out": outcome(graph, ids, target, dist, g1), "g2_out": outcome(graph, ids, target, dist, g2),
            "same_guess": g1 is not None and g1 == g2, "same_sample": picks[0] is not None and picks[0] == picks[1],
            "text_s1": sing[0], "text_s2": sing[1], "text_top2": t2,
        })
    out = Path(a.out or f"data/runs/probe-top2-{int(time.time())}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    d = pd.DataFrame(rows)
    d["dist"] = d.dist.clip(upper=max(a.dists))
    c = lambda col: d[col] == "closer"  # noqa: E731
    d["single_1"], d["single_either"] = c("s1_out"), c("s1_out") | c("s2_out")
    d["top2_1st"], d["top2_2nd"], d["top2_either"] = c("g1_out"), c("g2_out"), c("g1_out") | c("g2_out")
    d["random2_either"] = 1 - (1 - d.chance_closer) ** 2  # two random links (with replacement), for scale
    d["single_invalid"] = (d.s1_out == "invalid") | (d.s2_out == "invalid")
    d["top2_invalid"] = (d.g1_out == "invalid") | (d.g2_out == "invalid")
    cols = ["single_1", "single_either", "same_sample", "top2_1st", "top2_2nd", "top2_either", "same_guess",
            "top2_format_ok", "single_invalid", "top2_invalid", "random2_either"]
    pd.set_option("display.width", 250)
    print((d.groupby("dist")[cols].mean() * 100).round(1).assign(n=d.groupby("dist").size()).to_string())
    print(f"rows in {out}")


def coverage(a, graph, meta, msgs, run) -> None:
    """K samples per position per temperature: does some sample move closer, how many links do they spread over."""
    temps = a.temperatures or [a.temperature]
    ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= a.coverage]
    rows = []
    for temp in temps:
        a.temperature = temp
        t0 = time.time()
        texts = run(msgs, a.coverage)
        print(f"T={temp}: {len(msgs) * a.coverage} answers in {time.time() - t0:.0f}s", flush=True)
        for (sid, s, ids, target, dist), samples in zip(meta, texts):
            picks = [parse_action(lenient_answer(t)) for t in samples]
            outs = [outcome(graph, ids, target, dist, p) for p in picks]
            row = {"temperature": temp, "state": sid, "dist": s["dist"], "target": s["target"], "path": s["path"],
                   "picks": picks, "outcomes": outs,
                   "invalid": sum(o == "invalid" for o in outs) / len(outs),
                   "closer": sum(o == "closer" for o in outs) / len(outs),
                   "distinct_valid": len({p for p, o in zip(picks, outs) if o != "invalid"}),
                   "chars": sum(map(len, samples)) / len(samples)}
            for k in ks:
                row[f"cov@{k}"] = any(o == "closer" for o in outs[:k])
            rows.append(row)
    out = Path(a.out or f"data/runs/probe-coverage-{int(time.time())}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    d = pd.DataFrame(rows)
    d["dist"] = d.dist.clip(upper=max(a.dists))
    cols = ["closer", "invalid"] + [f"cov@{k}" for k in ks]
    pd.set_option("display.width", 250)
    t = d.groupby(["dist", "temperature"])[cols].mean() * 100
    t["distinct_valid"] = d.groupby(["dist", "temperature"]).distinct_valid.mean()
    t["chars"] = d.groupby(["dist", "temperature"]).chars.mean()
    print(t.round(1).to_string())
    print(f"rows in {out}")


if __name__ == "__main__":
    main()
