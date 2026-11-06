"""Baselines: random / oracle / human / trained model side by side, per hop distance, with error bars."""
import plotly.express as px
import streamlit as st

from wikigame.webapp import st_helpers as H
from wikigame.webapp.analysis import (baseline_table, comparability_warnings, has_hop_results,
                                      latest_per_policy, policy_label)
from wikigame.webapp.reference import CAVEATS, PAPER_ENGLISH

st.title("⚖️ Baseline comparison")
H.refresh_button()

eligible = [r for r in H.runs() if has_hop_results(r)]
if not eligible:
    st.info("No runs with per-hop eval results yet. Run `uv run python scripts/smoke_run.py`.")
    st.stop()

with st.sidebar:
    graphs = sorted({r.graph or "unknown" for r in eligible})
    graph = st.selectbox("Graph", graphs, help="Only runs on the same graph (and eval pairs) are comparable.")
    latest = st.toggle("Latest run per agent only", value=True, help="Hides duplicate re-runs of the same baseline.")
pool = [r for r in eligible if (r.graph or "unknown") == graph]
if latest:
    pool = latest_per_policy(pool)

names = sorted({f"{policy_label(r)} · {r.machine}" for r in pool})
picked = st.multiselect("Agents", names, default=names)
runs = [r for r in pool if f"{policy_label(r)} · {r.machine}" in picked]
if not runs:
    st.stop()
for w in comparability_warnings(runs):
    st.warning(w)

tbl = baseline_table(runs)
tbl["agent"] = tbl.agent + " · " + tbl.machine
tbl["hops"] = tbl.hop.astype(str)
ORDER = {"oracle": 0, "human": 1, "random": 3}
agent_order = sorted(tbl.agent.unique(), key=lambda a: (ORDER.get(a.split(" · ")[0], 2), a))

st.subheader("Success rate by hop distance")
fig = px.bar(tbl, x="hops", y="success", color="agent", barmode="group", text_auto=".0%",
             error_y="err_plus", error_y_minus="err_minus", category_orders={"agent": agent_order},
             labels={"hops": "hops to target", "success": "success rate"}, hover_data={"n": True, "ci_low": ":.2f", "ci_high": ":.2f"})
fig.update_yaxes(range=[0, 1.08], tickformat=".0%")
fig.update_layout(legend={"orientation": "h", "y": -0.25})
st.plotly_chart(fig, width="stretch")
st.caption("Error bars: 95% Wilson interval. Wide bars mean few episodes behind that bar (typical for human games).")

left, right = st.columns(2)
with left:
    st.subheader("Extra steps on successes")
    ex = tbl.dropna(subset=["extra_steps"])
    if ex.empty:
        st.caption("No agent succeeded often enough to measure.")
    else:
        f = px.bar(ex, x="hops", y="extra_steps", color="agent", barmode="group", category_orders={"agent": agent_order},
                   labels={"hops": "hops to target", "extra_steps": "steps above optimal"})
        f.update_layout(showlegend=False)
        st.plotly_chart(f, width="stretch")
with right:
    st.subheader("Success vs difficulty")
    f = px.line(tbl, x="hop", y="success", color="agent", markers=True, category_orders={"agent": agent_order})
    f.update_xaxes(dtick=1, title="hops to target")
    f.update_yaxes(range=[0, 1.05], tickformat=".0%")
    f.update_layout(showlegend=False)
    st.plotly_chart(f, width="stretch")

st.subheader("Numbers")
st.dataframe(tbl[["agent", "hop", "success", "ci_low", "ci_high", "extra_steps", "n", "run"]], hide_index=True, width="stretch",
             column_config={"success": st.column_config.ProgressColumn("success", min_value=0, max_value=1, format="%.2f"),
                            "ci_low": st.column_config.NumberColumn("95% low", format="%.2f"),
                            "ci_high": st.column_config.NumberColumn("95% high", format="%.2f"),
                            "extra_steps": st.column_config.NumberColumn("extra steps", format="%.2f")})
if not any(a.startswith("human") for a in tbl.agent):
    st.caption("No human baseline yet: play a few games on the **Play** page and upload them.")

with st.expander("📚 Reference: the paper's results on English Wikipedia (rough orientation only)"):
    st.warning(CAVEATS)
    ref = px.bar(x=[v[0] for v in PAPER_ENGLISH.values()], y=list(PAPER_ENGLISH), orientation="h", text_auto=".0%",
                 error_x=[v[1] for v in PAPER_ENGLISH.values()], labels={"x": "success rate (100-step limit)", "y": ""})
    ref.update_xaxes(range=[0, 1], tickformat=".0%")
    ref.update_layout(height=320, margin={"t": 10})
    st.plotly_chart(ref, width="stretch")
    st.caption("Source: arXiv 2610.13948 Table 4, English column. ⚠️ These values were read through a model-extracted "
               "summary of the table, not verbatim, so verify against the paper before relying on them. "
               "GPT-o3 mini and the human success rate are omitted because they could not be confirmed.")
