"""Pure analysis helpers (no Streamlit): episodes/runs -> DataFrames, path comparison, subgraphs.

Episodes are the dicts produced by ``Episode.to_dict()`` (loaded from the W&B episodes artifact).
"""
from __future__ import annotations

import random
import re

import pandas as pd

from ..env import classify_invalid
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


CAUSE_INFO = {
    "target_not_linked": ("Names the target (not linked)", "The model typed the target article, but it isn't linked "
                          "from the current page. It knows where it wants to go and ignores the link list."),
    "not_linked": ("Real page, not linked", "A real article title that is not in the current link list."),
    "unknown_title": ("Not an article title", "A title that doesn't exist in the graph: hallucinated or misspelled."),
    "current_page": ("Re-picks current page", "The model chose the page it is already on."),
    "revisit": ("Revisits a page", "The model went back to an earlier page on its path (not allowed)."),
    "unparseable": ("Unparseable output", "No valid <answer>{json}</answer>: a format failure (e.g. ```json fences)."),
    "unrecorded": ("Cause not recorded", "Old episode without a stored cause and no local graph to infer it from."),
}


def step_causes(graph: WikiGraph | None, ep: dict) -> list[str | None]:
    """Invalid-action cause per step (None for valid steps). Uses the stored cause when present; episodes
    recorded before causes existed are backfilled from the path (needs the graph), else 'unrecorded'."""
    out: list[str | None] = []
    visited: list[str] = [ep["path"][0]]
    cur = ep["path"][0]
    target = graph.id_of(ep["target"]) if graph else None
    for s in ep["steps"]:
        if s["valid"]:
            out.append(None)
            cur = s["chosen"]
            visited.append(cur)
            continue
        if "cause" in s and s["cause"]:
            out.append(s["cause"])
        elif graph is not None and target is not None and graph.id_of(cur) is not None:
            vis = {graph.id_of(t) for t in visited}
            out.append(classify_invalid(graph, graph.id_of(cur), vis, target, s["chosen"]) or "unrecorded")
        else:
            out.append("unrecorded")
    return out


def steps_df(episodes: list[dict], graph: WikiGraph | None = None) -> pd.DataFrame:
    """One row per action (valid or not) with format compliance of the raw model text and, for invalid
    actions, the cause (see step_causes)."""
    rows = []
    for i, ep in enumerate(episodes):
        causes = step_causes(graph, ep)
        for j, s in enumerate(ep["steps"], 1):
            text = s.get("text") or ""
            think = THINK_RE.search(text)
            rows.append({"ep_idx": i, "n_hops": ep.get("n_hops"), "step": j, "valid": s["valid"],
                         "chosen": s["chosen"], "reward": s["reward"], "n_links": s["n_links"],
                         "format_ok": bool(FORMAT_RE.search(text)) and parse_action(text) is not None,
                         "think_chars": len(think.group(1).strip()) if think else 0, "cause": causes[j - 1]})
    return pd.DataFrame(rows, columns=["ep_idx", "n_hops", "step", "valid", "chosen", "reward", "n_links",
                                       "format_ok", "think_chars", "cause"])


def invalid_cause_table(steps: pd.DataFrame) -> pd.DataFrame:
    """Invalid actions per cause: count, share of invalid actions, share of ALL actions. Largest first."""
    bad = steps[~steps.valid]
    if bad.empty:
        return pd.DataFrame(columns=["cause", "label", "count", "share_of_invalid", "share_of_actions"])
    t = bad.groupby("cause").size().rename("count").reset_index()
    t["label"] = t.cause.map(lambda c: CAUSE_INFO.get(c, (c, ""))[0])
    t["share_of_invalid"] = t["count"] / len(bad)
    t["share_of_actions"] = t["count"] / len(steps)
    return t.sort_values("count", ascending=False).reset_index(drop=True)[
        ["cause", "label", "count", "share_of_invalid", "share_of_actions"]]


def top_bad_titles(steps: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Most frequent invalid answers with their cause (unparseable output has no title and is skipped)."""
    bad = steps[(~steps.valid) & steps.chosen.notna()]
    if bad.empty:
        return pd.DataFrame(columns=["chosen", "cause", "count"])
    return (bad.groupby(["chosen", "cause"]).size().rename("count").reset_index()
            .sort_values("count", ascending=False).head(k).reset_index(drop=True))


def reward_breakdown(episodes: list[dict], reward_cfg: dict | None = None) -> pd.DataFrame:
    """Per-episode reward split into the env's components. ``residual`` should be ~0: if it is not, the
    run used a different reward config than the one given (or the env changed)."""
    rc = {**vars(RewardConfig()), **(reward_cfg or {})}
    rows = []
    for i, e in enumerate(episodes):
        calls = e["n_valid"] + e["n_invalid"]
        shaped = rc["shaping"] > 0
        parts = {"success bonus": rc["success"] if e["success"] else 0.0,
                 "step cost": -rc["step_cost"] * calls,
                 "invalid penalty": sum((rc["invalid_by_cause"].get(st.get("cause"), rc["shaped_invalid"]) if shaped
                                        else rc["invalid"]) for st in e["steps"] if not st["valid"]),
                 "dead-end penalty": rc["dead_end"] if e["end_reason"] == "dead_end" else 0.0}
        if shaped:
            parts["progress shaping"] = sum(st.get("shaping_reward", 0.0) for st in e["steps"])
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


# ======================================================================================================
# Batch C: baseline comparison
# ======================================================================================================


def wilson_interval(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; sensible for small n and p near 0 or 1."""
    if not n:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def policy_label(run: RunInfo) -> str:
    """Human-readable agent name: the baseline's policy ('random', 'oracle', 'human') or the experiment."""
    if run.job_type in ("baseline", "human"):
        return str(run.config.get("policy", run.group or run.name))
    return run.group or run.name


def has_hop_results(run: RunInfo) -> bool:
    return any(HOP_METRIC.match(k) and HOP_METRIC.match(k)["name"] == "success" for k in run.summary)


def latest_per_policy(runs: list[RunInfo]) -> list[RunInfo]:
    """Keep the newest run for each (agent, machine, graph): hides accidental duplicate runs."""
    best: dict[tuple, RunInfo] = {}
    for r in runs:
        k = (policy_label(r), r.machine, r.graph)
        if k not in best or r.created_at > best[k].created_at:
            best[k] = r
    return list(best.values())


def hop_n(run: RunInfo, hop: int) -> float | None:
    """Episodes behind success@hop: logged as eval/n@hop; older runs fall back to config.n_per_hop."""
    n = run.summary.get(f"eval/n@{hop}")
    return n if n is not None else run.config.get("n_per_hop")


def baseline_table(runs: list[RunInfo]) -> pd.DataFrame:
    """Long frame: agent, machine, hop, success (+95% CI), extra_steps, n."""
    rows = []
    for r in runs:
        for k in sorted(int(HOP_METRIC.match(x)["hop"]) for x in r.summary if HOP_METRIC.match(x) and HOP_METRIC.match(x)["name"] == "success"):
            s = r.summary[f"eval/success@{k}"]
            n = hop_n(r, k)
            lo, hi = wilson_interval(s * n, n) if n else (float("nan"), float("nan"))
            rows.append({"agent": policy_label(r), "machine": r.machine, "run": r.name, "hop": k, "success": s,
                         "ci_low": lo, "ci_high": hi, "err_plus": hi - s if n else None,
                         "err_minus": s - lo if n else None, "extra_steps": r.summary.get(f"eval/extra_steps@{k}"), "n": n})
    return pd.DataFrame(rows, columns=["agent", "machine", "run", "hop", "success", "ci_low", "ci_high",
                                       "err_plus", "err_minus", "extra_steps", "n"])


def comparability_warnings(runs: list[RunInfo]) -> list[str]:
    """Things that make a side-by-side comparison misleading."""
    out = []
    for key, label in (("graph", "graphs"), ("max_steps", "step limits")):
        vals = {r.config.get(key) for r in runs}
        if len(vals) > 1:
            out.append(f"Selected runs use different {label}: {sorted(map(str, vals))}. Success rates are not directly comparable.")
    n_per = {r.config.get("n_per_hop") for r in runs if r.config.get("n_per_hop")}
    if len(n_per) > 1:
        out.append(f"Different numbers of episodes per hop {sorted(n_per)}: error bars differ accordingly.")
    return out
