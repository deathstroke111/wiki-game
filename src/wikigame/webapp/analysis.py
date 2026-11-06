"""Pure analysis helpers (no Streamlit): episodes/runs -> DataFrames, path comparison, subgraphs.

Episodes are the dicts produced by ``Episode.to_dict()`` (loaded from the W&B episodes artifact).
"""
from __future__ import annotations

import random
import re

import pandas as pd

from ..graph import WikiGraph
from .data import RunInfo

HOP_METRIC = re.compile(r"^eval/(?P<name>[a-z_]+)@(?P<hop>\d+)$")


def runs_df(runs: list[RunInfo]) -> pd.DataFrame:
    rows = []
    for r in runs:
        rows.append({
            "run": r.name, "id": r.id, "machine": r.machine, "job_type": r.job_type, "group": r.group,
            "state": r.state, "created": r.created_at, "graph": r.graph or "",
            "success": r.summary.get("eval/success"), "path_len": r.summary.get("eval/path_len"),
            "invalid_rate": r.summary.get("eval/invalid_rate"), "reward": r.summary.get("eval/reward"),
        })
    return pd.DataFrame(rows)


def hop_metric_df(runs: list[RunInfo], name: str = "success") -> pd.DataFrame:
    """Long frame (run, machine, hop, value) from summary keys like 'eval/success@3'."""
    rows = []
    for r in runs:
        for k, v in r.summary.items():
            m = HOP_METRIC.match(k)
            if m and m["name"] == name:
                rows.append({"run": r.name, "machine": r.machine, "hop": int(m["hop"]), "value": v})
    df = pd.DataFrame(rows, columns=["run", "machine", "hop", "value"])
    return df.sort_values(["run", "hop"]).reset_index(drop=True)


def episodes_df(episodes: list[dict]) -> pd.DataFrame:
    rows = []
    for i, e in enumerate(episodes):
        rows.append({
            "idx": i, "n_hops": e.get("n_hops"), "source": e["source"], "target": e["target"],
            "success": e["success"], "end_reason": e["end_reason"], "optimal_steps": e["optimal_steps"],
            "extra_steps": e.get("extra_steps"), "path_len": len(e["path"]) - 1,
            "n_invalid": e["n_invalid"], "total_reward": e["total_reward"],
            "label": f"{e['source']} → {e['target']}",
        })
    return pd.DataFrame(rows)


def optimal_path(graph: WikiGraph, ep: dict) -> list[str] | None:
    s, t = graph.id_of(ep["source"]), graph.id_of(ep["target"])
    if s is None or t is None:
        return None
    p = graph.shortest_path(s, t)
    return None if p is None else [graph.titles[i] for i in p]


def step_progress(graph: WikiGraph, ep: dict) -> list[dict]:
    """Per valid move: hop distance to the target before/after, and whether it was a wrong turn
    (the move did not get strictly closer). Moves line up with consecutive pairs in ep['path']."""
    t = graph.id_of(ep["target"])
    d = graph.dist_to(t) if t is not None else {}
    rows = []
    for i, (a, b) in enumerate(zip(ep["path"], ep["path"][1:])):
        ia, ib = graph.id_of(a), graph.id_of(b)
        before, after = d.get(ia), d.get(ib)
        delta = None if before is None or after is None else before - after
        rows.append({"step": i + 1, "from": a, "to": b, "dist_before": before, "dist_after": after,
                     "delta": delta, "wrong_turn": delta is None or delta <= 0})
    return rows


def path_subgraph(graph: WikiGraph, ep: dict, n_neighbors: int = 4, seed: int = 0) -> dict:
    """Nodes/edges to draw: agent path, optimal path, and a few sampled out-neighbours of each path node.

    Returns {"nodes": {title: {"kinds": set, "wrong": bool}}, "edges": [(u, v, kind)]} with edge kinds
    'agent' | 'optimal' | 'neighbor'; node kinds 'source' | 'target' | 'agent' | 'optimal' | 'neighbor'.
    """
    rng = random.Random(seed)
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], str] = {}

    def add(title: str, kind: str) -> None:
        nodes.setdefault(title, {"kinds": set(), "wrong": False})["kinds"].add(kind)

    agent, opt = ep["path"], optimal_path(graph, ep) or []
    for t in agent:
        add(t, "agent")
    for t in opt:
        add(t, "optimal")
    add(ep["source"], "source")
    add(ep["target"], "target")
    for a, b in zip(opt, opt[1:]):
        edges[(a, b)] = "optimal"
    for a, b in zip(agent, agent[1:]):
        edges[(a, b)] = "agent"  # agent wins over optimal when the edge is shared

    on_path = set(nodes)
    for t in set(agent) | set(opt):
        i = graph.id_of(t)
        if i is None:
            continue
        outs = [graph.titles[j] for j in graph.adj[i] if graph.titles[j] not in on_path]
        for o in rng.sample(outs, min(n_neighbors, len(outs))):
            add(o, "neighbor")
            edges.setdefault((t, o), "neighbor")

    for row in step_progress(graph, ep):
        if row["wrong_turn"]:
            nodes[row["to"]]["wrong"] = True
    return {"nodes": nodes, "edges": [(u, v, k) for (u, v), k in edges.items()]}


def outcome_label(ep_row: pd.Series) -> str:
    return "✅ success" if ep_row["success"] else f"❌ {ep_row['end_reason']}"
