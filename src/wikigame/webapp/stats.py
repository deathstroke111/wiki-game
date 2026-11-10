"""Statistics across agents (no Streamlit): paired game-by-game tests, training trends, failure anatomy.

Every agent plays the same held-out games, so comparisons are paired on the game (hop, source, target).
Each agent played each game once, so sampling noise of a stochastic policy is not separated from skill.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from ..graph import WikiGraph
from .analysis import step_causes, wilson_interval

GAME_KEYS = ["n_hops", "source", "target"]

_STAGE = re.compile(r"stage(?P<num>\d+)(?P<sub>[a-z]?)-(?:step(?P<step>\d+)|final)$")


# ---- agent ordering --------------------------------------------------------------------------------------
def ladder_key(agent: str) -> tuple | None:
    """Training order of an R1 agent: the untrained model first, then stage, sub-stage, step (final last).
    None for agents that are not on the R1 training ladder."""
    if agent.startswith("zeroshot-r1"):
        return (0, "", 0)
    m = _STAGE.search(agent)
    if not m:
        return None
    return (int(m["num"]), m["sub"], int(m["step"]) if m["step"] else math.inf)


def agent_sort_key(agent: str) -> tuple:
    """Display order: random, zero-shot Qwen, the R1 ladder, then oracle."""
    key = ladder_key(agent)
    if key is not None:
        return (2, key, agent)
    if "oracle" in agent:
        return (3, (), agent)
    return (0 if "random" in agent else 1, (), agent)


def planned_pairs(agents: list[str]) -> list[tuple[str, str]]:
    """The comparisons fixed in advance, as (later, earlier): each ladder step, final vs untrained, and
    final vs zero-shot Qwen (lenient parsing, like the R1 evals)."""
    ladder = sorted((a for a in agents if ladder_key(a) is not None), key=ladder_key)
    if not ladder:
        return []
    pairs = [(b, a) for a, b in zip(ladder, ladder[1:])]
    if len(ladder) > 2:
        pairs.append((ladder[-1], ladder[0]))
    pairs += [(ladder[-1], a) for a in agents if "qwen" in a and a.endswith("-lenient")]
    return pairs


# ---- paired game-by-game statistics -------------------------------------------------------------------------
def success_matrix(agents: dict[str, list[dict]]) -> pd.DataFrame:
    """Games x agents, 1/0 for success and NaN where the agent did not play the game. A game drawn more
    than once for an agent (possible on tiny graphs) counts its first attempt only."""
    cols = {}
    for name, eps in agents.items():
        keys = [(e.get("n_hops"), e["source"], e["target"]) for e in eps]
        s = pd.Series([float(e["success"]) for e in eps], index=pd.MultiIndex.from_tuples(keys, names=GAME_KEYS))
        cols[name] = s[~s.index.duplicated(keep="first")]
    return pd.DataFrame(cols).sort_index()


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar test: under 'no difference' each discordant game is a fair coin."""
    n = a_only + b_only
    return 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)


def holm(pvalues) -> np.ndarray:
    """Holm-adjusted p-values (family-wise error control, valid under any dependence between tests)."""
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((p.size - np.arange(p.size)) * p[order]).clip(max=1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def bootstrap_diff_ci(diffs: np.ndarray, strata: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                      level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of the mean of paired differences, resampling games within each stratum (hop)."""
    rng = np.random.default_rng(seed)
    total = np.zeros(n_boot)
    for s in np.unique(strata):
        d = diffs[strata == s]
        total += rng.choice(d, size=(n_boot, d.size)).sum(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(total / diffs.size, [tail, 1 - tail])
    return float(low), float(high)


def compare(wide: pd.DataFrame, a: str, b: str, hop: int | None = None, seed: int = 0) -> dict:
    """Agent a vs b on the games both played (optionally one hop distance)."""
    d = wide[[a, b]].dropna()
    if hop is not None:
        d = d[d.index.get_level_values("n_hops") == hop]
    x, y = d[a].to_numpy(), d[b].to_numpy()
    a_only, b_only = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    row = {"a": a, "b": b, "hop": "all" if hop is None else hop, "games": len(d), "a_rate": float("nan"),
           "b_rate": float("nan"), "diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
           "a_only": a_only, "b_only": b_only, "p": 1.0}
    if len(d):
        low, high = bootstrap_diff_ci(x - y, d.index.get_level_values("n_hops").to_numpy(), seed=seed)
        row.update(a_rate=float(x.mean()), b_rate=float(y.mean()), diff=float((x - y).mean()), ci_low=low, ci_high=high,
                   p=mcnemar_exact(a_only, b_only))
    return row


def comparison_table(wide: pd.DataFrame, pairs: list[tuple[str, str]], hop: int | None = None) -> pd.DataFrame:
    """One row per pair with a Holm-adjusted p-value across the whole family of pairs."""
    cols = ["a", "b", "hop", "games", "a_rate", "b_rate", "diff", "ci_low", "ci_high", "a_only", "b_only", "p", "p_holm"]
    if not pairs:
        return pd.DataFrame(columns=cols)
    t = pd.DataFrame([compare(wide, a, b, hop) for a, b in pairs])
    t["p_holm"] = holm(t.p)
    return t[cols]


def conditional_logit(wide: pd.DataFrame, agents: list[str], reference: str) -> pd.DataFrame:
    """Odds of winning vs the reference agent, with each game as its own stratum. Game difficulty (and so
    hop distance) is conditioned out, and only games the agents split carry information."""
    from statsmodels.discrete.conditional_models import ConditionalLogit

    d = wide[agents].dropna().reset_index(drop=True)
    long = d.rename_axis("game").reset_index().melt(id_vars="game", var_name="agent", value_name="won")
    x = pd.get_dummies(long.agent, dtype=float).drop(columns=reference)[[a for a in agents if a != reference]]
    fit = ConditionalLogit(long.won.astype(float), x, groups=long.game).fit(disp=0)
    ci = np.asarray(fit.conf_int())
    return pd.DataFrame({"agent": list(x.columns), "odds_ratio": np.exp(np.asarray(fit.params)),
                         "or_low": np.exp(ci[:, 0]), "or_high": np.exp(ci[:, 1]), "p": np.asarray(fit.pvalues),
                         "games": len(d)})


def games_needed(delta: float, discordant: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Games for a paired comparison to detect a success-rate change `delta`, given the share of games the
    two agents split (normal approximation to McNemar)."""
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return math.ceil(((za * math.sqrt(discordant) + zb * math.sqrt(discordant - delta ** 2)) / delta) ** 2)


def success_by_agent(agents: dict[str, list[dict]]) -> pd.DataFrame:
    """Success rate with a Wilson 95% interval, per agent and hop (hop 'all' = pooled)."""
    rows = []
    for name, eps in agents.items():
        hops = sorted({e.get("n_hops") for e in eps})
        for hop in ["all", *hops]:
            sub = [e for e in eps if hop == "all" or e.get("n_hops") == hop]
            wins = sum(e["success"] for e in sub)
            low, high = wilson_interval(wins, len(sub))
            rows.append({"agent": name, "hop": str(hop), "games": len(sub), "wins": wins, "success": wins / len(sub),
                         "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


# ---- training process ----------------------------------------------------------------------------------------
STEP_COL = "train/global_step"


def per_step(history: pd.DataFrame, metric: str) -> pd.Series:
    """One value per optimizer step (W&B history also holds rows that lack this metric)."""
    if metric not in history or STEP_COL not in history:
        return pd.Series(dtype=float)
    d = history[[STEP_COL, metric]].dropna()
    return d.groupby(STEP_COL)[metric].mean()


def first_last_change(history: pd.DataFrame, metrics: dict[str, str], window: int = 50) -> pd.DataFrame:
    """Mean over the first vs last `window` steps with a normal-approximation 95% CI of the change."""
    rows = []
    for metric, label in metrics.items():
        s = per_step(history, metric)
        if len(s) < 2 * window:
            continue
        first, last = s.iloc[:window], s.iloc[-window:]
        se = math.sqrt(first.var(ddof=1) / window + last.var(ddof=1) / window)
        diff = float(last.mean() - first.mean())
        rows.append({"metric": label, "first": float(first.mean()), "last": float(last.mean()), "change": diff,
                     "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se})
    return pd.DataFrame(rows, columns=["metric", "first", "last", "change", "ci_low", "ci_high"])


def block_means(history: pd.DataFrame, metric: str, block: int = 25) -> pd.DataFrame:
    """Metric averaged over blocks of optimizer steps (per-step values are very noisy)."""
    s = per_step(history, metric)
    if s.empty:
        return pd.DataFrame(columns=["step", "value"])
    steps = (((s.index.to_numpy().astype(int) - 1) // block) + 1) * block
    return s.groupby(steps).mean().rename("value").rename_axis("step").reset_index()


# ---- failure anatomy -----------------------------------------------------------------------------------------
OUTCOMES = ["closer", "sideways", "farther", "invalid"]


def action_table(graph: WikiGraph, episodes: list[dict]) -> pd.DataFrame:
    """One row per model answer: BFS distance before it, what it did, and the chance that a uniformly
    random unvisited link would have moved closer from that page."""
    cols = ["ep_idx", "n_hops", "step", "dist_before", "outcome", "cause", "repeat_invalid", "chance_closer", "success"]
    rows = []
    distances: dict[str, dict[int, int]] = {}
    for i, ep in enumerate(episodes):
        target = graph.id_of(ep["target"])
        if target is None:
            continue
        dist = distances.setdefault(ep["target"], graph.dist_to(target))
        here = graph.id_of(ep["path"][0])
        visited = {here}
        previous_bad = None
        for j, (step, cause) in enumerate(zip(ep["steps"], step_causes(graph, ep)), 1):
            before = dist.get(here)
            links = [n for n in graph.adj[here] if n not in visited]
            chance = None
            if before is not None and links:
                chance = sum(dist.get(n, math.inf) < before for n in links) / len(links)
            repeat = False
            if step["valid"]:
                to = graph.id_of(step["chosen"])
                after = dist.get(to)
                if before is None or after is None or after > before:
                    outcome = "farther"
                else:
                    outcome = "closer" if after < before else "sideways"
                here = to
                visited.add(to)
                previous_bad = None
            else:
                outcome = "invalid"
                repeat = previous_bad is not None and previous_bad == step["chosen"]
                previous_bad = step["chosen"]
            rows.append({"ep_idx": i, "n_hops": ep.get("n_hops"), "step": j, "dist_before": before, "outcome": outcome,
                         "cause": cause, "repeat_invalid": repeat, "chance_closer": chance, "success": ep["success"]})
    return pd.DataFrame(rows, columns=cols)


def game_table(graph: WikiGraph, episodes: list[dict]) -> pd.DataFrame:
    """One row per game: start, closest and final BFS distance to the target."""
    rows = []
    distances: dict[str, dict[int, int]] = {}
    for i, ep in enumerate(episodes):
        target = graph.id_of(ep["target"])
        if target is None:
            continue
        dist = distances.setdefault(ep["target"], graph.dist_to(target))
        along = [dist.get(graph.id_of(t)) for t in ep["path"]]
        known = [d for d in along if d is not None]
        rows.append({"ep_idx": i, "n_hops": ep.get("n_hops"), "success": ep["success"], "end_reason": ep["end_reason"],
                     "start_dist": along[0], "best_dist": min(known) if known else None, "final_dist": along[-1],
                     "n_valid": ep["n_valid"], "n_invalid": ep["n_invalid"]})
    return pd.DataFrame(rows)


def outcome_by_distance(actions: pd.DataFrame, max_dist: int = 4) -> pd.DataFrame:
    """Per distance from the target: answer outcome shares, and move quality against random-link chance.
    Distances above max_dist are pooled into the last row."""
    cols = ["dist", "answers", *OUTCOMES, "closer_if_valid", "chance_closer", "lift"]
    a = actions.dropna(subset=["dist_before"])
    if a.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for dist, g in a.groupby(a.dist_before.clip(upper=max_dist).astype(int)):
        valid = g[g.outcome != "invalid"]
        closer = float((valid.outcome == "closer").mean()) if len(valid) else float("nan")
        chance = float(valid.chance_closer.mean()) if len(valid) else float("nan")
        rows.append({"dist": f"{dist}+" if dist == max_dist else str(dist), "answers": len(g),
                     **{o: float((g.outcome == o).mean()) for o in OUTCOMES},
                     "closer_if_valid": closer, "chance_closer": chance, "lift": closer / chance if chance else float("nan")})
    return pd.DataFrame(rows, columns=cols)


def behaviour_summary(actions: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Headline behaviour of one agent, counted over all of its answers and games."""
    bad = actions[actions.outcome == "invalid"]
    valid = actions[actions.outcome != "invalid"]
    fails = games[~games.success]
    return {"games": len(games), "answers": len(actions), "invalid": len(bad) / len(actions) if len(actions) else float("nan"),
            "repeat_share_of_invalid": float(bad.repeat_invalid.mean()) if len(bad) else 0.0,
            "closer_if_valid": float((valid.outcome == "closer").mean()) if len(valid) else float("nan"),
            "chance_closer": float(valid.chance_closer.mean()) if len(valid) else float("nan"),
            "failed_games": len(fails),
            "fail_end_invalid": float((fails.end_reason == "invalid_action").mean()) if len(fails) else float("nan"),
            "fail_end_step_limit": float((fails.end_reason == "step_limit").mean()) if len(fails) else float("nan"),
            "fail_final_dist": float(fails.final_dist.mean()) if len(fails) else float("nan"),
            "fail_one_click_away": float((fails.final_dist == 1).mean()) if len(fails) else float("nan"),
            "fail_got_closer": float((fails.best_dist < fails.start_dist).mean()) if len(fails) else float("nan")}
