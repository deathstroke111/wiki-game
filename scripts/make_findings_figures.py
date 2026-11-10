"""Render the figures of docs/findings/ from W&B episodes and training history (needs `wandb login`).

  uv run python scripts/make_findings_figures.py

The sampled Qwen rerun is read from a local JSONL, which is optional (its figure is skipped when missing).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from wikigame.webapp import stats as S
from wikigame.webapp.analysis import CAUSE_INFO, episodes_df, has_hop_results, latest_per_policy, policy_label, steps_df, wilson_interval
from wikigame.webapp.data import WandbSource

OUT = Path("docs/findings/figures")
GRAPH = None  # set in main(), used by outcome_probs
QWEN_SAMPLED = next(iter(sorted(Path("data/runs").glob("zeroshot-qwen2.5-1.5b-instruct-lenient-T0.6-*.jsonl"))), None)

Q, R, S1, S2A, S2, S3 = ("zeroshot-qwen2.5-1.5b-instruct-lenient", "zeroshot-r1-1.5b-lenient", "grpo-step-r1-stage1b-final",
                         "grpo-step-r1-stage2-step100", "grpo-step-r1-stage2-final", "grpo-step-r1-stage3-final")
NAMES = {Q: "Qwen 1.5B zero-shot", R: "R1 1.5B untrained", S1: "R1 stage 1b", S2A: "R1 stage 2 (step 100)", S2: "R1 stage 2", S3: "R1 stage 3"}
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
AGENT_COLORS = {NAMES[Q]: BLUE, NAMES[R]: ORANGE, NAMES[S3]: AQUA, NAMES[S2]: YELLOW}
OUTCOME_COLORS = dict(zip(S.OUTCOMES, [BLUE, ORANGE, AQUA, YELLOW]))
GRID, INK, MUTED, GRAY = "#e6e5e0", "#0b0b0b", "#52514e", "#9a9992"
TRAIN_RUNS = (("kkqk572s", "stage 1b"), ("sa7pbhpc", "stage 2"), ("mg1m4f44", "stage 3"))


def styled(fig: go.Figure, title: str, height: int = 480, width: int = 1000, legend: bool = True, facets: bool = False) -> go.Figure:
    fig.update_layout(template="plotly_white", title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 20, "color": INK}},
                      font={"family": "Helvetica, Arial, sans-serif", "size": 14, "color": MUTED}, width=width, height=height,
                      margin={"l": 70, "r": 30, "t": 130 if facets and legend else 90, "b": 70}, plot_bgcolor="white", paper_bgcolor="white", showlegend=legend,
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.09 if facets else 1.0, "x": 0, "title": None})
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font={"size": 15, "color": INK}))
    return fig


def save(fig: go.Figure, name: str) -> None:
    fig.write_image(OUT / f"{name}.png", scale=2)
    print("wrote", name)


def labels_above_intervals(fig: go.Figure, rates: pd.DataFrame, agents: list[str]) -> go.Figure:
    """Grouped-bar value labels placed just above each interval, so they never sit on the error bar."""
    hops = sorted(rates.hop.unique())
    width = 0.8 / len(agents)
    for r in rates.itertuples():
        x = hops.index(r.hop) + (agents.index(r.agent) - (len(agents) - 1) / 2) * width
        fig.add_annotation(x=x, y=r.ci_high, text=f"{r.success:.0%}", showarrow=False, yshift=12, font={"size": 13, "color": MUTED})
    return fig


def key(e: dict) -> tuple:
    return (e["n_hops"], e["source"], e["target"])


def outcome_probs(episodes: list[dict]) -> np.ndarray:
    """P(closer, sideways, farther, invalid) per distance 1, 2, 3, 4+, with a small smoothing count."""
    a = S.action_table(GRAPH, episodes).dropna(subset=["dist_before"])
    a = a.assign(d=a.dist_before.clip(upper=4).astype(int))
    t = pd.crosstab(a.d, a.outcome).reindex(index=[1, 2, 3, 4], columns=S.OUTCOMES, fill_value=0).to_numpy(float) + 0.5
    return t / t.sum(axis=1, keepdims=True)


def simulate(p: np.ndarray, hops=(1, 2, 3, 4), n: int = 40_000, seed: int = 0) -> float:
    """Mean success across hops when every answer is drawn from p, under the game's own limits."""
    rng = np.random.default_rng(seed)
    rates = []
    for hop in hops:
        d, steps, invalid = np.full(n, hop), np.zeros(n, int), np.zeros(n, int)
        alive, won = np.ones(n, bool), np.zeros(n, bool)
        for _ in range(40):
            idx = np.where(alive)[0]
            if not idx.size:
                break
            out = (rng.random((idx.size, 1)) > p[np.minimum(d[idx], 4) - 1].cumsum(axis=1)).sum(axis=1)
            bad = idx[out == 3]
            invalid[bad] += 1
            alive[bad] = ~((invalid[bad] >= 3) | (steps[bad] + invalid[bad] >= 10))
            good, moves = idx[out != 3], out[out != 3]
            steps[good] += 1
            d[good] = np.minimum(d[good] + np.array([-1, 0, 1])[moves], 6)
            finished = good[d[good] == 0]
            won[finished], alive[finished] = True, False
            alive[good[(d[good] > 0) & (steps[good] >= 10)]] = False
        rates.append(won.mean())
    return float(np.mean(rates))


def with_invalid(p: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    moves = p[:, :3] / p[:, :3].sum(axis=1, keepdims=True)
    return np.c_[moves * (1 - invalid)[:, None], invalid]


def boost_closer(p: np.ndarray, factor: float) -> np.ndarray:
    """Multiply P(closer) at 2+ clicks out, taking the mass from sideways moves first and then farther ones."""
    p = p.copy()
    for i in (1, 2, 3):
        extra = min(p[i, 0] * (factor - 1), p[i, 1] + p[i, 2])
        side = min(extra, p[i, 1])
        p[i, 0] += extra
        p[i, 1] -= side
        p[i, 2] -= extra - side
    return p


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = WandbSource()
    global GRAPH
    graph = GRAPH = src.load_graph("wikispeedia")
    runs = latest_per_policy([r for r in src.list_runs() if r.job_type in ("eval", "baseline") and has_hop_results(r)])
    A = {policy_label(r): src.load_episodes(r.id, src.episode_sets(r.id)[-1].key) for r in runs}
    shared = {key(e) for e in A[R]}  # the 80 games every R1 stage played
    on_shared = {a: [e for e in eps if key(e) in shared] for a, eps in A.items()}
    three = [Q, R, S3]
    actions = {a: S.action_table(graph, A[a]) for a in three}

    # 01 success by hop
    rates = S.success_by_agent({NAMES[a]: A[a] for a in three})
    rates = rates[rates.hop != "all"].assign(plus=lambda d: d.ci_high - d.success, minus=lambda d: d.success - d.ci_low)
    fig = px.bar(rates, x="hop", y="success", color="agent", barmode="group", error_y="plus", error_y_minus="minus",
                 color_discrete_map=AGENT_COLORS, category_orders={"agent": [NAMES[a] for a in three]}, labels={"hop": "shortest path to the target (clicks)", "success": "games won"})
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%")
    fig.update_traces(marker_line={"width": 2, "color": "white"})
    labels_above_intervals(fig, rates, [NAMES[a] for a in three])
    save(styled(fig, "Success rate by game difficulty, with 95% intervals"), "01_success_by_hop")

    # 02 forest plot of the planned paired comparisons
    wide = S.success_matrix({a: A[a] for a in (Q, R, S1, S2A, S2, S3)})
    t = S.comparison_table(wide, S.planned_pairs([R, S1, S2A, S2, S3, Q]))
    t["label"] = t.a.map(NAMES) + " vs " + t.b.map(NAMES) + "  (n=" + t.games.astype(str) + ")"
    fig = px.scatter(t, x="diff", y="label", error_x=t.ci_high - t["diff"], error_x_minus=t["diff"] - t.ci_low, labels={"diff": "change in success rate, same games", "label": ""})
    fig.update_traces(marker={"size": 12, "color": AQUA})
    fig.add_vline(x=0, line_width=1.5, line_dash="dot", line_color=GRAY)
    fig.update_xaxes(tickformat="+.0%", range=[-0.15, 0.5])
    fig.update_yaxes(autorange="reversed")
    fig = styled(fig, "What each training step changed, compared game by game", height=440, legend=False)
    fig.update_layout(margin={"l": 330})
    save(fig, "02_paired_comparisons")

    # 03 invalid causes
    rows = []
    for a in three:
        total = len(actions[a])
        for c in ("target_not_linked", "unparseable", "unknown_title", "current_page", "revisit", "not_linked"):
            rows.append({"agent": NAMES[a], "cause": CAUSE_INFO[c][0], "share": (actions[a].cause == c).sum() / total})
    d = pd.DataFrame(rows)
    fig = px.bar(d, y="cause", x="share", color="agent", barmode="group", orientation="h", text_auto=".1%", color_discrete_map=AGENT_COLORS,
                 labels={"share": "share of all answers", "cause": ""})
    fig.update_xaxes(tickformat=".0%", range=[0, 0.78])
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line={"width": 2, "color": "white"})
    fig = styled(fig, "Why answers were invalid", height=600)
    fig.update_layout(margin={"l": 210})
    save(fig, "03_invalid_causes")

    # 04 valid moves in failed games
    frames = []
    for a in (Q, S3):
        df = episodes_df(A[a])
        frames.append(df[~df.success].assign(agent=NAMES[a]))
    fails = pd.concat(frames, ignore_index=True)
    counts = fails.groupby(["agent", "path_len"]).size().rename("n").reset_index()
    counts["share"] = counts.n / counts.groupby("agent").n.transform("sum")
    fig = px.bar(counts, x="path_len", y="share", facet_col="agent", color="agent", color_discrete_map=AGENT_COLORS, text_auto=".0%",
                 category_orders={"agent": [NAMES[Q], NAMES[S3]]}, labels={"path_len": "valid moves before the game ended", "share": "share of failed games"})
    fig.update_xaxes(dtick=1, range=[-0.6, 10.6])
    fig.update_yaxes(tickformat=".0%", range=[0, 0.8])
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line={"width": 2, "color": "white"})
    save(styled(fig, "How long failed games lasted", height=460, legend=False), "04_failed_game_length")

    # 05 how games end
    rows = []
    for a in (Q, S3):
        for hop, g in episodes_df(A[a]).groupby("n_hops"):
            for reason, n in g.end_reason.value_counts().items():
                rows.append({"agent": NAMES[a], "hop": str(hop), "ending": reason.replace("_", " "), "share": n / len(g)})
    fig = px.bar(pd.DataFrame(rows), x="hop", y="share", color="ending", facet_col="agent", text_auto=".0%", category_orders={"agent": [NAMES[Q], NAMES[S3]], "ending": ["success", "step limit", "invalid action", "dead end"]},
                 color_discrete_map={"success": AQUA, "step limit": YELLOW, "invalid action": ORANGE, "dead end": BLUE}, labels={"hop": "shortest path to the target (clicks)", "share": "share of games"})
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_traces(marker_line={"width": 2, "color": "white"})
    save(styled(fig, "How games ended", height=520, facets=True), "05_how_games_end")

    # 06 reward of failed games
    counts = fails.assign(reward=fails.total_reward.round(2)).groupby(["agent", "reward"]).size().rename("n").reset_index()
    counts["share"] = counts.n / counts.groupby("agent").n.transform("sum")
    fig = px.bar(counts, x="reward", y="share", facet_col="agent", color="agent", color_discrete_map=AGENT_COLORS, category_orders={"agent": [NAMES[Q], NAMES[S3]]},
                 labels={"reward": "total episode reward (unshaped)", "share": "share of failed games"})
    fig.update_traces(width=0.016)
    fig.update_yaxes(tickformat=".0%", range=[0, 0.75])
    save(styled(fig, "Reward of failed games", height=440, legend=False), "06_failed_game_reward")

    # 07 reasoning length
    rows = [steps_df(on_shared[a], graph).assign(agent=NAMES[a]) for a in (R, S2, S3)]
    think = pd.concat(rows, ignore_index=True).query("think_chars > 0")
    think = think.assign(bin=(think.think_chars // 250) * 250 + 125)
    binned = think.groupby(["agent", "bin"]).size().rename("n").reset_index()
    binned["share"] = binned.n / binned.groupby("agent").n.transform("sum")
    fig = px.bar(binned, x="bin", y="share", facet_col="agent", color="agent", color_discrete_map=AGENT_COLORS, category_orders={"agent": [NAMES[R], NAMES[S2], NAMES[S3]]},
                 labels={"bin": "characters inside <think>", "share": "share of closed think blocks"})
    fig.update_traces(width=220)
    fig.update_xaxes(range=[0, 4900], dtick=1000)
    fig.update_yaxes(tickformat=".0%", range=[0, 0.32])
    save(styled(fig, "Reasoning length on the held-out games (closed think blocks only)", height=420, legend=False), "07_reasoning_length")

    # 08 and 09 training curves
    histories = {name: src.run_history(rid) for rid, name in TRAIN_RUNS}
    stage_colors = dict(zip((n for _, n in TRAIN_RUNS), (BLUE, ORANGE, AQUA)))

    def curves(metric: str) -> pd.DataFrame:
        frames, offset = [], 0
        for name, h in histories.items():
            b = S.block_means(h, metric)
            frames.append(b.assign(step=b.step + offset, stage=name))
            offset = int(frames[-1].step.max())
        return pd.concat(frames, ignore_index=True)

    fig = px.line(curves("train/completions/mean_length"), x="step", y="value", color="stage", markers=True, color_discrete_map=stage_colors,
                  labels={"step": "cumulative training step", "value": "mean completion length (tokens)"})
    fig.update_traces(line={"width": 2}, marker={"size": 8})
    fig.update_yaxes(range=[0, 900])
    fig.add_hline(y=512, line_width=1.5, line_dash="dot", line_color=GRAY, annotation_text="512-token cap (stages 2 and 3)", annotation_position="top right")
    save(styled(fig, "Completion length during training", height=440), "08_training_completion_length")

    metrics = {"train/env/invalid_rate": "invalid answers", "train/env/closer_rate": "moves that got closer"}
    long = pd.concat([curves(m).assign(metric=label) for m, label in metrics.items()], ignore_index=True)
    fig = px.line(long, x="step", y="value", color="stage", facet_col="metric", markers=True, color_discrete_map=stage_colors, category_orders={"metric": list(metrics.values())},
                  labels={"step": "cumulative training step", "value": "share of training answers"})
    fig.update_traces(line={"width": 2}, marker={"size": 7})
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    save(styled(fig, "Training metrics, averaged over 25 steps (states get harder at each boundary)", height=500, facets=True), "09_training_invalid_and_closer")

    # 10 what the agent does at each distance
    rows = [S.outcome_by_distance(actions[a]).melt(id_vars="dist", value_vars=S.OUTCOMES, var_name="outcome", value_name="share").assign(agent=NAMES[a]) for a in three]
    fig = px.bar(pd.concat(rows, ignore_index=True), x="dist", y="share", color="outcome", facet_col="agent", text_auto=".0%", color_discrete_map=OUTCOME_COLORS, category_orders={"agent": [NAMES[a] for a in three], "outcome": S.OUTCOMES, "dist": ["1", "2", "3", "4+"]},
                 labels={"dist": "clicks from the target", "share": "share of answers"})
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_traces(marker_line={"width": 2, "color": "white"}, textfont={"size": 11})
    save(styled(fig, "What the agent does at each distance from the target", height=560, width=1200, facets=True), "10_distance_outcomes")

    # 11 move quality against chance
    rows = []
    for a in three:
        for r in S.outcome_by_distance(actions[a]).itertuples():
            rows.append({"series": NAMES[a], "dist": r.dist, "rate": r.closer_if_valid})
    for r in S.outcome_by_distance(actions[S3]).itertuples():
        rows.append({"series": "Random link", "dist": r.dist, "rate": r.chance_closer})
    colors = {**AGENT_COLORS, "Random link": GRAY}
    fig = px.bar(pd.DataFrame(rows), x="dist", y="rate", color="series", barmode="group", text_auto=".0%", color_discrete_map=colors, category_orders={"series": ["Random link", NAMES[Q], NAMES[R], NAMES[S3]], "dist": ["1", "2", "3", "4+"]},
                 labels={"dist": "clicks from the target", "rate": "valid moves that got closer"})
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line={"width": 2, "color": "white"})
    save(styled(fig, "Move quality against picking a random link", height=480), "11_move_quality_vs_chance")

    # 12 extra steps
    rows = []
    for a in (Q, S3):
        ok = episodes_df(A[a]).query("success")
        for hop, g in ok.groupby("n_hops"):
            rows.append({"agent": NAMES[a], "hop": f"{hop} click" + ("s" if hop > 1 else ""), "extra": g.extra_steps.mean(), "label": f"{g.extra_steps.mean():.2f}  (n={len(g)})"})
    fig = px.bar(pd.DataFrame(rows), x="hop", y="extra", color="agent", barmode="group", text="label", color_discrete_map=AGENT_COLORS, labels={"hop": "shortest path to the target", "extra": "extra moves above optimal, won games"})
    fig.update_yaxes(range=[0, 4])
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line={"width": 2, "color": "white"})
    save(styled(fig, "Detours in won games", height=440), "12_extra_steps")

    # 13 what-if scenarios from a simulation calibrated on stage 3's per-distance behaviour
    p = outcome_probs(A[S3])
    scen = pd.DataFrame({"scenario": ["Stage 3 as measured (simulated)", "Half the invalid answers", "No invalid answers", "1.5x more closer moves (2+ clicks out)",
                                      "2x more closer moves (2+ clicks out)", "2x closer moves and no invalid answers"],
                         "success": [simulate(p), simulate(with_invalid(p, p[:, 3] / 2)), simulate(with_invalid(p, np.zeros(4))), simulate(boost_closer(p, 1.5)),
                                     simulate(boost_closer(p, 2)), simulate(with_invalid(boost_closer(p, 2), np.zeros(4)))]})
    print(scen.round(3).to_string())
    fig = px.bar(scen, y="scenario", x="success", orientation="h", text_auto=".0%", labels={"success": "predicted success across hops 1-4", "scenario": ""})
    fig.update_traces(marker_color=AQUA, textposition="outside", cliponaxis=False)
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    fig = styled(fig, "Simulated what-ifs for stage 3", height=440, legend=False)
    fig.update_layout(margin={"l": 340})
    save(fig, "13_counterfactuals")

    # 14 greedy vs sampled Qwen against stage 3
    if QWEN_SAMPLED:
        sampled = [json.loads(line) for line in QWEN_SAMPLED.read_text().splitlines() if line]
        by = {"Qwen greedy": A[Q], "Qwen sampled (T=0.6)": sampled, NAMES[S3]: A[S3]}
        rates = S.success_by_agent(by)
        rates = rates[rates.hop != "all"].assign(plus=lambda d: d.ci_high - d.success, minus=lambda d: d.success - d.ci_low)
        fig = px.bar(rates, x="hop", y="success", color="agent", barmode="group", error_y="plus", error_y_minus="minus", color_discrete_map={"Qwen greedy": BLUE, "Qwen sampled (T=0.6)": YELLOW, NAMES[S3]: AQUA},
                     category_orders={"agent": list(by)}, labels={"hop": "shortest path to the target (clicks)", "success": "games won"})
        fig.update_yaxes(range=[0, 1.12], tickformat=".0%")
        fig.update_traces(marker_line={"width": 2, "color": "white"})
        labels_above_intervals(fig, rates, list(by))
        save(styled(fig, "Same model, same games: does sampling change Qwen's score?"), "14_qwen_decoding")


if __name__ == "__main__":
    main()
