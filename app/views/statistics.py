"""Statistics: how much each training stage really changed, how training went, and how games fail."""
import pandas as pd
import plotly.express as px
import streamlit as st

from wikigame.webapp import st_helpers as H
from wikigame.webapp import stats as S
from wikigame.webapp.analysis import has_hop_results, latest_per_policy, policy_label

OUTCOME_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, one per S.OUTCOMES
TRAIN_METRICS = {"train/env/invalid_rate": "invalid rate", "train/env/closer_rate": "closer rate",
                 "train/env/success_rate": "per-step success", "train/reward": "reward",
                 "train/completions/mean_length": "completion length", "train/frac_reward_zero_std": "zero-std groups"}


@st.cache_data(ttl=600, show_spinner="Loading episodes from W&B…")
def load_agents(run_ids: tuple[str, ...]) -> dict[str, list[dict]]:
    out = {}
    for rid in run_ids:
        sets = H.episode_sets(rid)
        if sets:
            out[rid] = H.episodes(rid, sets[-1].key)
    return out


@st.cache_data(ttl=600, show_spinner="Loading training history…")
def load_history(run_id: str) -> pd.DataFrame:
    return H.history(run_id)


@st.cache_data(show_spinner="Replaying games on the graph…")
def game_tables(run_id: str, _graph, _episodes) -> tuple[pd.DataFrame, pd.DataFrame]:
    return S.action_table(_graph, _episodes), S.game_table(_graph, _episodes)


@st.cache_data(show_spinner="Bootstrapping…")
def compare_cached(_wide, wide_key: str, pairs: tuple, hop) -> pd.DataFrame:
    return S.comparison_table(_wide, list(pairs), hop)


def short(agent: str) -> str:
    return agent.replace("grpo-step-r1-", "").replace("zeroshot-", "").replace("-1.5b", "").replace("-instruct", "")


def rate_cell(row) -> str:
    return f"{row.success:.0%} [{row.ci_low:.0%}–{row.ci_high:.0%}]"


def verdict(row) -> str:
    if row.p_holm < 0.05:
        return "holds after correction"
    if row.p < 0.05:
        return "suggestive only"
    return "borderline" if row.p < 0.10 else "no clear evidence"


st.title("📐 Statistics")
st.caption("Every agent played the same held-out games, so agents are compared game by game.")
H.refresh_button()

eligible = latest_per_policy([r for r in H.runs() if r.job_type in ("eval", "baseline") and has_hop_results(r)])
if not eligible:
    st.info("No eval or baseline runs with per-hop results yet.")
    st.stop()

with st.sidebar:
    graph_name = st.selectbox("Graph", sorted({r.graph or "unknown" for r in eligible}))
pool = {policy_label(r): r for r in eligible if (r.graph or "unknown") == graph_name}
loaded = load_agents(tuple(sorted(r.id for r in pool.values())))
pool = {name: r for name, r in pool.items() if r.id in loaded}
if not pool:
    st.info("None of these runs logged episodes.")
    st.stop()
names = sorted(pool, key=S.agent_sort_key)
with st.sidebar:
    picked = st.multiselect("Agents", names, default=names, format_func=short)
if not picked:
    st.info("Pick at least one agent in the sidebar.")
    st.stop()
episodes = {a: loaded[pool[a].id] for a in picked}
graph = H.graph(pool[picked[0]].graph)

with st.expander("How to read these numbers", icon=":material/info:"):
    st.markdown(
        "- **One attempt per game.** Sampled agents (T=0.6) would score differently on a re-run, and that noise "
        "cannot be separated from skill here.\n"
        "- **Uneven sample sizes.** Each comparison only uses games both agents played; check its `games` column.\n"
        "- **Settings differ.** Zero-shot Qwen used greedy decoding and a 256-token cap, and R1 sampled at T=0.6 "
        "with a 1024-token cap, so read that comparison as a rough one.\n"
        "- **Multiple testing.** Holm-adjusted p-values account for testing the planned comparisons together.")

section = st.segmented_control("Section", ["Stage comparison", "Training process", "Failure anatomy"],
                               default="Stage comparison", key="stats_section") or "Stage comparison"

# ---- stage comparison ----------------------------------------------------------------------------------------
if section == "Stage comparison":
    wide = S.success_matrix(episodes)
    wide_key = ",".join(picked) + str(wide.shape)

    st.subheader("Success rate with 95% Wilson intervals")
    rates = S.success_by_agent(episodes)
    rates["cell"] = rates.apply(rate_cell, axis=1)
    table = rates.pivot(index="agent", columns="hop", values="cell").reindex(picked)
    table.insert(0, "games", [len(episodes[a]) for a in picked])
    st.dataframe(table.rename(index=short), width="stretch")

    st.subheader("Planned comparisons")
    pairs = S.planned_pairs(picked)
    if not pairs:
        st.info("Planned comparisons need at least two agents from the R1 training ladder.")
    else:
        hops = ["All hops", *sorted(int(h) for h in wide.index.get_level_values("n_hops").unique())]
        hop_choice = st.segmented_control("Hop distance", hops, default="All hops", key="stats_hop") or "All hops"
        hop = None if hop_choice == "All hops" else hop_choice
        t = compare_cached(wide, wide_key, tuple(pairs), hop)
        t = t[t.games > 0].copy()
        t["comparison"] = t.a.map(short) + " vs " + t.b.map(short)
        t["verdict"] = t.apply(verdict, axis=1)
        forest = px.scatter(t, x="diff", y="comparison", error_x=t.ci_high - t["diff"], error_x_minus=t["diff"] - t.ci_low,
                            labels={"diff": "change in success rate (first minus second)", "comparison": ""})
        forest.update_traces(marker={"size": 10, "color": OUTCOME_COLORS[0]})
        forest.add_vline(x=0, line_width=1, line_dash="dot")
        forest.update_xaxes(tickformat="+.0%")
        forest.update_yaxes(autorange="reversed")
        forest.update_layout(height=120 + 45 * len(t), margin={"t": 10})
        st.plotly_chart(forest, width="stretch")
        st.dataframe(t[["comparison", "games", "a_rate", "b_rate", "diff", "ci_low", "ci_high", "a_only", "b_only",
                        "p", "p_holm", "verdict"]], hide_index=True, width="stretch",
                     column_config={"a_rate": st.column_config.NumberColumn("first", format="percent"),
                                    "b_rate": st.column_config.NumberColumn("second", format="percent"),
                                    "diff": st.column_config.NumberColumn("change", format="percent"),
                                    "ci_low": st.column_config.NumberColumn("95% low", format="percent"),
                                    "ci_high": st.column_config.NumberColumn("95% high", format="percent"),
                                    "a_only": st.column_config.NumberColumn("only first won"),
                                    "b_only": st.column_config.NumberColumn("only second won"),
                                    "p": st.column_config.NumberColumn("exact p", format="%.3f"),
                                    "p_holm": st.column_config.NumberColumn("Holm p", format="%.3f")})
        st.caption("Intervals resample games within each hop. The exact test uses only games the two agents split.")

        ladder = sorted((a for a in picked if S.ladder_key(a) is not None), key=S.ladder_key)
        latest_step = pairs[len(ladder) - 2]  # planned_pairs lists the ladder steps first, in training order
        if len(ladder) >= 2:
            st.subheader("Each stage against the untrained model, controlling for the game")
            with st.spinner("Fitting…"):
                logit = S.conditional_logit(wide, ladder, ladder[0])
            logit["agent"] = logit.agent.map(short)
            st.dataframe(logit, hide_index=True, width="stretch",
                         column_config={"odds_ratio": st.column_config.NumberColumn("odds ratio", format="%.1f"),
                                        "or_low": st.column_config.NumberColumn("95% low", format="%.1f"),
                                        "or_high": st.column_config.NumberColumn("95% high", format="%.1f"),
                                        "p": st.column_config.NumberColumn(format="%.4f")})
            st.caption(f"Conditional logistic regression on the games all of these agents played, relative to "
                       f"`{short(ladder[0])}`. Intervals are wide because few games separate the agents.")

        st.subheader("Games needed to confirm a change")
        last = S.compare(wide, *latest_step)
        if last["games"] and last["a_only"] + last["b_only"]:
            discordant = (last["a_only"] + last["b_only"]) / last["games"]
            needed = pd.DataFrame([{"change to detect": f"{d:.0%}", "games (paired)": S.games_needed(d, discordant)}
                                   for d in (0.05, 0.10, 0.15, 0.20) if d ** 2 < discordant])
            st.dataframe(needed, hide_index=True)
            st.caption(f"80% power at a 5% significance level, assuming agents split {discordant:.0%} of games as "
                       f"{short(last['a'])} and {short(last['b'])} do.")

    st.subheader("Compare any two agents")
    if len(picked) >= 2:
        left, right = st.columns(2)
        a = left.selectbox("First agent", picked, index=len(picked) - 1, format_func=short)
        b = right.selectbox("Second agent", picked, index=0, format_func=short)
        if a == b:
            st.info("Pick two different agents.")
        else:
            row = S.compare(wide, a, b)
            st.write(f"{row['games']} shared games: {row['a_rate']:.1%} vs {row['b_rate']:.1%}, change {row['diff']:+.1%} "
                     f"[{row['ci_low']:+.1%}, {row['ci_high']:+.1%}]. Only first won {row['a_only']}, only second won "
                     f"{row['b_only']}, exact p = {row['p']:.3f}. Exploratory: not corrected for multiple testing.")

# ---- training process ----------------------------------------------------------------------------------------
elif section == "Training process":
    train_runs = sorted((r for r in H.runs() if r.job_type == "train"), key=lambda r: r.created_at)
    if not train_runs:
        st.info("No training runs in this project yet.")
        st.stop()

    def stage_of(run) -> str:
        return next((t.split(":", 1)[1] for t in run.tags if t.startswith("stage:")), run.name)

    histories = {r.id: load_history(r.id) for r in train_runs}
    metric = st.selectbox("Metric", list(TRAIN_METRICS), format_func=TRAIN_METRICS.get)
    frames, offset = [], 0
    for r in train_runs:
        blocks = S.block_means(histories[r.id], metric)
        if not blocks.empty:
            frames.append(blocks.assign(step=blocks.step + offset, stage=stage_of(r)))
            offset = int(frames[-1].step.max())
    if frames:
        curve = px.line(pd.concat(frames, ignore_index=True), x="step", y="value", color="stage", markers=True,
                        color_discrete_sequence=OUTCOME_COLORS, labels={"step": "cumulative training step", "value": TRAIN_METRICS[metric]})
        curve.update_traces(line={"width": 2})
        curve.update_layout(margin={"t": 10}, legend={"orientation": "h", "y": -0.25})
        st.plotly_chart(curve, width="stretch")
        st.caption("Averages over 25 steps. Training states got harder between stages, so a level shift at a stage "
                   "boundary is not a change in skill.")

    st.subheader("Change within each stage (first vs last 50 steps)")
    changes = [S.first_last_change(histories[r.id], TRAIN_METRICS).assign(stage=stage_of(r)) for r in train_runs]
    changes = pd.concat(changes, ignore_index=True)
    if changes.empty:
        st.caption("No stage has at least 100 steps of metrics.")
    else:
        st.dataframe(changes[["stage", "metric", "first", "last", "change", "ci_low", "ci_high"]], hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("first", "last", "change", "ci_low", "ci_high")})
        st.caption("Intervals treat steps as independent, which makes them somewhat too narrow.")

    st.subheader("Training metrics next to held-out results")
    rows = []
    for r in train_runs:
        stage = stage_of(r)
        end = S.first_last_change(histories[r.id], TRAIN_METRICS)
        late = dict(zip(end.metric, end["last"])) if not end.empty else {}
        match = next((a for a in names if a.endswith(f"stage{stage}-final")), None)
        held = pool[match].summary if match else {}
        rows.append({"stage": stage, "steps": len(S.per_step(histories[r.id], "train/env/invalid_rate")),
                     "token cap": r.config.get("max_completion"), "train hops": str(r.config.get("hops")),
                     "hop mix": str(r.config.get("mix")), "train invalid": late.get("invalid rate"),
                     "train step success": late.get("per-step success"), "held-out invalid": held.get("eval/invalid_rate"),
                     "held-out success": held.get("eval/success")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent")
                                for c in ("train invalid", "train step success", "held-out invalid", "held-out success")})
    st.caption("Held-out sample sizes differ by stage. With so few stages this is a side-by-side view, not a correlation.")

# ---- failure anatomy -----------------------------------------------------------------------------------------
else:
    if graph is None:
        st.warning(f"Failure anatomy needs the local graph `data/{graph_name}.json`.")
        st.stop()
    tables = {a: game_tables(pool[a].id, graph, episodes[a]) for a in picked}

    st.subheader("Behaviour by agent")
    summary = pd.DataFrame({short(a): S.behaviour_summary(*tables[a]) for a in picked}).T
    st.dataframe(summary, width="stretch",
                 column_config={"games": st.column_config.NumberColumn(format="%d"),
                                "answers": st.column_config.NumberColumn(format="%d"),
                                "failed_games": st.column_config.NumberColumn("failed games", format="%d"),
                                "invalid": st.column_config.NumberColumn(format="percent"),
                                "repeat_share_of_invalid": st.column_config.NumberColumn("invalid: same as last", format="percent"),
                                "closer_if_valid": st.column_config.NumberColumn("closer (valid moves)", format="percent"),
                                "chance_closer": st.column_config.NumberColumn("closer by chance", format="percent"),
                                "fail_end_invalid": st.column_config.NumberColumn("fails: 3 invalid", format="percent"),
                                "fail_end_step_limit": st.column_config.NumberColumn("fails: step limit", format="percent"),
                                "fail_final_dist": st.column_config.NumberColumn("fails: final distance", format="%.2f"),
                                "fail_one_click_away": st.column_config.NumberColumn("fails: one click away", format="percent"),
                                "fail_got_closer": st.column_config.NumberColumn("fails: ever got closer", format="percent")})
    st.caption("Pooled rates depend on which states an agent reached, so compare move quality per distance below.")

    default_agent = next((a for a in reversed(picked) if S.ladder_key(a) is not None), picked[-1])
    agent = st.selectbox("Agent", picked, index=picked.index(default_agent), format_func=short)
    actions, games = tables[agent]

    st.subheader("What the agent does at each distance from the target")
    by_dist = S.outcome_by_distance(actions)
    if by_dist.empty:
        st.info("No answers recorded for this agent on this graph.")
        st.stop()
    long = by_dist.melt(id_vars="dist", value_vars=S.OUTCOMES, var_name="outcome", value_name="share")
    bars = px.bar(long, x="dist", y="share", color="outcome", text_auto=".0%", color_discrete_sequence=OUTCOME_COLORS,
                  category_orders={"outcome": S.OUTCOMES, "dist": list(by_dist.dist)},
                  labels={"dist": "clicks from the target (shortest path)", "share": "share of answers"})
    bars.update_yaxes(range=[0, 1], tickformat=".0%")
    bars.update_traces(marker_line={"width": 2, "color": "rgba(0,0,0,0)"})
    bars.update_layout(legend={"orientation": "h", "y": -0.25}, margin={"t": 10})
    st.plotly_chart(bars, width="stretch")
    st.dataframe(by_dist, hide_index=True, width="stretch",
                 column_config={**{o: st.column_config.NumberColumn(format="percent") for o in S.OUTCOMES},
                                "closer_if_valid": st.column_config.NumberColumn("closer (valid moves)", format="percent"),
                                "chance_closer": st.column_config.NumberColumn("closer by chance", format="percent"),
                                "lift": st.column_config.NumberColumn("lift over chance", format="%.2f")})
    st.caption("Lift compares the agent's valid moves with picking a random unvisited link on the same pages. "
               "1.0 means no better than chance.")

    left, right = st.columns(2)
    with left:
        st.subheader("Invalid causes by distance")
        bad = actions[actions.outcome == "invalid"].dropna(subset=["dist_before"])
        if bad.empty:
            st.caption("This agent gave no invalid answers.")
        else:
            st.dataframe(pd.crosstab(bad.cause, bad.dist_before.clip(upper=4).astype(int)), width="stretch")
    with right:
        st.subheader("How games ended")
        st.dataframe(pd.crosstab(games.n_hops, games.end_reason), width="stretch")
