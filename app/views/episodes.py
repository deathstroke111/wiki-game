"""Episodes: browse a run's evaluated episodes, compare agent vs optimal path, see the path on the graph."""
import pandas as pd
import streamlit as st

from wikigame.webapp import st_helpers as H
from wikigame.webapp.analysis import episodes_df, optimal_path, path_subgraph, step_progress
from wikigame.webapp.viz import path_figure

st.title("🔍 Episodes")
H.refresh_button()

run, eset, episodes, graph = H.pick_episode_set()
df = episodes_df(episodes)

# ---- filters --------------------------------------------------------------------------------------
with st.sidebar:
    hops = st.multiselect("Hops to target", sorted(df.n_hops.dropna().unique()), default=sorted(df.n_hops.dropna().unique()))
    outcomes = st.multiselect("End reason", sorted(df.end_reason.unique()), default=sorted(df.end_reason.unique()))
    sort = st.selectbox("Sort by", ["extra_steps (worst first)", "path_len (longest first)", "total_reward (lowest first)", "original order"])
view = df[df.n_hops.isin(hops) & df.end_reason.isin(outcomes)]
view = {
    "extra_steps (worst first)": lambda d: d.assign(_k=d.extra_steps.fillna(99)).sort_values("_k", ascending=False),
    "path_len (longest first)": lambda d: d.sort_values("path_len", ascending=False),
    "total_reward (lowest first)": lambda d: d.sort_values("total_reward"),
    "original order": lambda d: d,
}[sort](view)

# ---- summary --------------------------------------------------------------------------------------
st.caption(f"`{run.name}` · machine **{run.machine}** · graph `{run.graph}`" + ("" if graph else " · ⚠️ graph file not found locally: showing text paths only"))
c = st.columns(5)
c[0].metric("Episodes", len(view))
c[1].metric("Success", f"{view.success.mean():.0%}" if len(view) else "–")
c[2].metric("Extra steps (successes)", f"{view.extra_steps.mean():.2f}" if view.extra_steps.notna().any() else "–")
c[3].metric("Mean path length", f"{view.path_len.mean():.1f}" if len(view) else "–")
c[4].metric("Invalid actions / ep", f"{view.n_invalid.mean():.2f}" if len(view) else "–")

if view.empty:
    st.info("No episodes match the filters.")
    st.stop()

table = view[["idx", "label", "n_hops", "success", "end_reason", "optimal_steps", "extra_steps", "path_len", "n_invalid", "total_reward"]]
event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                     selection_mode="single-row", key="ep_table",
                     column_config={"total_reward": st.column_config.NumberColumn(format="%.2f")})
rows = event.selection.rows
linked = st.query_params.get("idx")  # deep link from the Failures page
if rows:
    idx = int(table.iloc[rows[0]].idx)
elif linked is not None and linked.isdigit() and int(linked) < len(episodes):
    idx = int(linked)
else:
    idx = int(table.iloc[0].idx)
    st.caption("Showing the first row; click a row in the table to inspect another episode.")
ep = episodes[idx]

# ---- detail ---------------------------------------------------------------------------------------
st.divider()
st.subheader(f"{ep['source']} → {ep['target']}")
st.markdown(f"**{'✅ success' if ep['success'] else '❌ ' + ep['end_reason']}** · optimal {ep['optimal_steps']} steps · "
            f"agent took {len(ep['path']) - 1} valid steps with {ep['n_invalid']} invalid · reward {ep['total_reward']:.2f}")

opt = optimal_path(graph, ep) if graph else None
left, right = st.columns(2)
left.markdown("**Agent path**")
left.markdown(" → ".join(f"`{t}`" for t in ep["path"]))
right.markdown("**Optimal path**" + ("" if opt else " (needs the local graph file)"))
if opt:
    right.markdown(" → ".join(f"`{t}`" for t in opt))

if graph:
    n_nb = st.slider("Neighbouring pages shown per path node", 0, 10, 3, key="n_nb")
    st.plotly_chart(path_figure(path_subgraph(graph, ep, n_neighbors=n_nb)), width="stretch")

st.markdown("**Step by step**")
prog = iter(step_progress(graph, ep)) if graph else None
cur, rows_out = ep["source"], []
for i, s in enumerate(ep["steps"], 1):
    row = {"#": i, "from": cur, "chose": s["chosen"], "valid": s["valid"], "links shown": s["n_links"], "reward": round(s["reward"], 3)}
    if s["valid"]:
        if prog:
            p = next(prog)
            row.update({"dist before": p["dist_before"], "dist after": p["dist_after"], "wrong turn": p["wrong_turn"]})
        cur = s["chosen"]
    rows_out.append(row)
st.dataframe(pd.DataFrame(rows_out), hide_index=True, width="stretch")
with st.expander("Raw model output per step"):
    for i, s in enumerate(ep["steps"], 1):
        st.markdown(f"**Step {i}**")
        st.code(s["text"] or "(empty)", language="text")
