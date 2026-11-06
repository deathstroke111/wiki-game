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
    return step_progress_with(graph, ep, graph.dist_to(t) if t is not None else {})


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


# ======================================================================================================
# Batch B: failure breakdown + behaviour diagnostics
# ======================================================================================================

from ..env import RewardConfig, parse_action  # noqa: E402

FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# thresholds for the heuristic flags on the Diagnostics page (shown to the user next to each flag)
THRESH = {"invalid_rate": 0.05, "format_ok": 0.95, "step_limit_share": 0.70, "invalid_end_share": 0.10,
          "transfer_ratio": 0.5, "slow_extra_steps": 1.5}


def drift_df(graph: WikiGraph, episodes: list[dict]) -> pd.DataFrame:
    """One row per valid move: did it get strictly closer to the target (true BFS distance)?"""
    dists: dict[str, dict[int, int]] = {}
    rows = []
    for i, ep in enumerate(episodes):
        t = graph.id_of(ep["target"])
        if t is None:
            continue
        d = dists.setdefault(ep["target"], graph.dist_to(t))
        for r in step_progress_with(graph, ep, d):
            rows.append({"ep_idx": i, "n_hops": ep.get("n_hops"), "success": ep["success"], **r})
    cols = ["ep_idx", "n_hops", "success", "step", "from", "to", "dist_before", "dist_after", "delta", "wrong_turn"]
    return pd.DataFrame(rows, columns=cols)


def step_progress_with(graph: WikiGraph, ep: dict, d: dict[int, int]) -> list[dict]:
    """Like step_progress but with a precomputed distance-to-target map (avoids a BFS per episode)."""
    rows = []
    for i, (a, b) in enumerate(zip(ep["path"], ep["path"][1:])):
        before, after = d.get(graph.id_of(a)), d.get(graph.id_of(b))
        delta = None if before is None or after is None else before - after
        rows.append({"step": i + 1, "from": a, "to": b, "dist_before": before, "dist_after": after,
                     "delta": delta, "wrong_turn": delta is None or delta <= 0})
    return rows


def steps_df(episodes: list[dict]) -> pd.DataFrame:
    """One row per action (valid or not) with format compliance of the raw model text."""
    rows = []
    for i, ep in enumerate(episodes):
        for j, s in enumerate(ep["steps"], 1):
            text = s.get("text") or ""
            think = THINK_RE.search(text)
            rows.append({"ep_idx": i, "n_hops": ep.get("n_hops"), "step": j, "valid": s["valid"],
                         "chosen": s["chosen"], "reward": s["reward"], "n_links": s["n_links"],
                         "format_ok": bool(FORMAT_RE.search(text)) and parse_action(text) is not None,
                         "think_chars": len(think.group(1).strip()) if think else 0})
    return pd.DataFrame(rows, columns=["ep_idx", "n_hops", "step", "valid", "chosen", "reward", "n_links",
                                       "format_ok", "think_chars"])


def reward_breakdown(episodes: list[dict], reward_cfg: dict | None = None) -> pd.DataFrame:
    """Per-episode reward split into the env's components. ``residual`` should be ~0: if it is not, the
    run used a different reward config than the one given (or the env changed)."""
    rc = {**vars(RewardConfig()), **(reward_cfg or {})}
    rows = []
    for i, e in enumerate(episodes):
        calls = e["n_valid"] + e["n_invalid"]
        parts = {"success bonus": rc["success"] if e["success"] else 0.0,
                 "step cost": -rc["step_cost"] * calls,
                 "invalid penalty": rc["invalid"] * e["n_invalid"],
                 "dead-end penalty": rc["dead_end"] if e["end_reason"] == "dead_end" else 0.0}
        rows.append({"ep_idx": i, "n_hops": e.get("n_hops"), "end_reason": e["end_reason"],
                     "success": e["success"], **parts, "total": e["total_reward"],
                     "residual": e["total_reward"] - sum(parts.values())})
    return pd.DataFrame(rows)


def end_reason_table(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of end_reason per n_hops (rows: hops, columns: reasons)."""
    return df.groupby(["n_hops", "end_reason"]).size().unstack(fill_value=0)


def wrong_turn_by_step(drift: pd.DataFrame) -> pd.DataFrame:
    """Share of moves that were wrong turns, by position in the path (step 1, 2, ...)."""
    if drift.empty:
        return pd.DataFrame(columns=["step", "share", "moves"])
    g = drift.groupby("step").wrong_turn.agg(["mean", "size"]).reset_index()
    return g.rename(columns={"mean": "share", "size": "moves"})


def first_wrong_turn(drift: pd.DataFrame) -> pd.Series:
    """Step index of each episode's first wrong turn (episodes with none are absent)."""
    w = drift[drift.wrong_turn]
    return w.groupby("ep_idx").step.min()


def worst_failures(df: pd.DataFrame, drift: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Failed episodes ranked by how badly they went: wrong-turn share, then path length."""
    fails = df[~df.success].copy()
    if fails.empty:
        return fails
    share = drift.groupby("ep_idx").wrong_turn.mean() if not drift.empty else pd.Series(dtype=float)
    fails["wrong_turn_share"] = fails.idx.map(share).fillna(0.0)
    return fails.sort_values(["wrong_turn_share", "path_len"], ascending=False).head(k)


def diagnostic_flags(df: pd.DataFrame, steps: pd.DataFrame) -> list[dict]:
    """Heuristic health checks on agent behaviour. status: 'ok' | 'warn' | 'info'."""
    flags: list[dict] = []

    def add(name, status, value, rule, detail):
        flags.append({"name": name, "status": status, "value": value, "rule": rule, "detail": detail})

    n_act = len(steps)
    inv = float((~steps.valid).mean()) if n_act else 0.0
    add("Invalid actions", "warn" if inv > THRESH["invalid_rate"] else "ok", f"{inv:.1%}",
        f"warn above {THRESH['invalid_rate']:.0%}",
        "Titles not in the link list, revisits, or unparseable output. High values mean the agent has not "
        "learned the action format or is ignoring the link list.")
    fmt = float(steps.format_ok.mean()) if n_act else 1.0
    add("Format compliance", "warn" if fmt < THRESH["format_ok"] else "ok", f"{fmt:.1%}",
        f"warn below {THRESH['format_ok']:.0%}",
        "Share of actions with a well-formed <think>…</think><answer>{json}</answer>.")
    fails = df[~df.success]
    if len(fails):
        sl = float((fails.end_reason == "step_limit").mean())
        add("Wandering to the step limit", "warn" if sl > THRESH["step_limit_share"] else "ok", f"{sl:.0%} of failures",
            f"warn above {THRESH['step_limit_share']:.0%}",
            "Failures that burn every step without committing to a direction. Expected for a random agent; "
            "for a trained one it suggests the policy lacks a sense of progress toward the target.")
        ie = float((fails.end_reason == "invalid_action").mean())
        add("Quitting via invalid actions", "warn" if ie > THRESH["invalid_end_share"] else "ok", f"{ie:.0%} of failures",
            f"warn above {THRESH['invalid_end_share']:.0%}",
            "Episodes truncated because the agent kept emitting invalid actions: a formatting problem, "
            "not a navigation problem.")
    pos = df[(~df.success) & (df.total_reward > 0)]
    add("Reward without success", "warn" if len(pos) else "ok", f"{len(pos)} episodes", "warn if any",
        "A failed episode earned positive reward: the reward has a leak the agent can exploit.")
    hops = sorted(df.n_hops.dropna().unique())
    if len(hops) >= 2:
        lo, hi = df[df.n_hops == hops[0]].success.mean(), df[df.n_hops == hops[-1]].success.mean()
        ratio = hi / lo if lo > 0 else float("nan")
        bad = lo > 0 and ratio < THRESH["transfer_ratio"]
        add("Curriculum transfer", "info" if bad else "ok", f"{hi:.0%} @ {int(hops[-1])} hops vs {lo:.0%} @ {int(hops[0])}",
            f"flag if ratio below {THRESH['transfer_ratio']}",
            "Success collapses as the required hop distance grows: consider more training at harder levels.")
    ex = df.extra_steps.dropna()
    if len(ex):
        m = float(ex.mean())
        add("Detours on successes", "info" if m > THRESH["slow_extra_steps"] else "ok", f"{m:.2f} extra steps",
            f"flag above {THRESH['slow_extra_steps']}",
            "Reaches the target but not efficiently: a larger step cost or shaped distance reward may help.")
    return flags
