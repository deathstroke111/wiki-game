"""Failures: why and where the agent fails, using true BFS distances to judge every move."""
import plotly.express as px
import streamlit as st

from wikigame.webapp import st_helpers as H
from wikigame.webapp.analysis import (drift_df, end_reason_table, episodes_df, first_wrong_turn,
                                      wrong_turn_by_step, worst_failures)

st.title("🧯 Failure breakdown")
H.refresh_button()
run, eset, episodes, graph = H.pick_episode_set()
df = episodes_df(episodes)
st.caption(f"`{run.name}` · machine **{run.machine}** · {eset.split} step {eset.step} · {len(df)} episodes")

# ---- 1. how episodes end --------------------------------------------------------------------------
st.subheader("How episodes end, by hop distance")
tbl = end_reason_table(df)
long = tbl.reset_index().melt(id_vars="n_hops", var_name="end_reason", value_name="episodes")
long["n_hops"] = long.n_hops.astype(int).astype(str)
share = st.toggle("Show share instead of counts", value=True)
if share:
    long["episodes"] = long.episodes / long.groupby("n_hops").episodes.transform("sum")
order = ["success", "step_limit", "dead_end", "invalid_action"]
fig = px.bar(long, x="n_hops", y="episodes", color="end_reason", barmode="stack",
             category_orders={"end_reason": order}, labels={"n_hops": "hops to target", "episodes": "share" if share else "episodes"},
             color_discrete_map={"success": "#00CC96", "step_limit": "#EF553B", "dead_end": "#AB63FA", "invalid_action": "#FFA15A"})
if share:
    fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig, width="stretch")

# ---- 2. drift (needs the graph) -------------------------------------------------------------------
if graph is None:
    st.warning("The graph file for this run isn't available locally, so move-level analysis (drift, wrong turns) is disabled.")
    st.stop()

drift = drift_df(graph, episodes)
st.subheader("Drift: how often does a move fail to get closer?")
c = st.columns(3)
c[0].metric("Wrong-turn rate (all moves)", f"{drift.wrong_turn.mean():.0%}" if len(drift) else "–",
            help="A move is a wrong turn if the true hop distance to the target did not strictly decrease.")
fails = drift[~drift.success]
c[1].metric("…in failed episodes", f"{fails.wrong_turn.mean():.0%}" if len(fails) else "–")
ok = drift[drift.success]
c[2].metric("…in successful episodes", f"{ok.wrong_turn.mean():.0%}" if len(ok) else "–")

left, right = st.columns(2)
with left:
    st.markdown("**Wrong-turn rate by position in the path**")
    w = wrong_turn_by_step(drift)
    if len(w):
        f = px.bar(w, x="step", y="share", hover_data=["moves"], labels={"step": "move number", "share": "wrong-turn rate"})
        f.update_yaxes(range=[0, 1.02], tickformat=".0%")
        st.plotly_chart(f, width="stretch")
with right:
    st.markdown("**Where the first wrong turn happens**")
    fw = first_wrong_turn(drift)
    if len(fw):
        f = px.histogram(fw.rename("first wrong turn at move #"), x="first wrong turn at move #", nbins=int(fw.max()))
        f.update_layout(yaxis_title="episodes", bargap=0.1)
        st.plotly_chart(f, width="stretch")
    else:
        st.caption("No wrong turns at all.")

# ---- 3. worst failures ----------------------------------------------------------------------------
st.subheader("Worst failures")
bad = worst_failures(df, drift, k=15)
if bad.empty:
    st.success("No failed episodes in this batch.")
    st.stop()
show = bad[["idx", "label", "n_hops", "end_reason", "optimal_steps", "path_len", "wrong_turn_share"]]
ev = st.dataframe(show, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                  column_config={"wrong_turn_share": st.column_config.ProgressColumn("wrong-turn share", min_value=0, max_value=1, format="%.0f%%")})
if ev.selection.rows:
    row = show.iloc[ev.selection.rows[0]]
    if st.button(f"Open “{row.label}” in the Episodes explorer →"):
        st.query_params.update({"run": run.id, "set": eset.key, "idx": str(int(row.idx))})
        st.switch_page("views/episodes.py")
else:
    st.caption("Select a row to jump to that episode in the explorer.")
