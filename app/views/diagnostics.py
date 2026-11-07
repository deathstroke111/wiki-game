"""Diagnostics: format/validity health, path efficiency, reward decomposition, reward-hacking flags."""
import plotly.express as px
import streamlit as st

from wikigame.webapp import st_helpers as H
from wikigame.webapp.analysis import (CAUSE_INFO, diagnostic_flags, episodes_df, invalid_cause_table, reward_breakdown,
                                      steps_df, top_bad_titles)

st.title("🩺 Behaviour diagnostics")
H.refresh_button()
run, eset, episodes, graph = H.pick_episode_set()
df, steps = episodes_df(episodes), steps_df(episodes, graph)
st.caption(f"`{run.name}` · machine **{run.machine}** · {eset.split} step {eset.step} · {len(df)} episodes · {len(steps)} actions")

# ---- flags ----------------------------------------------------------------------------------------
st.subheader("Health checks")
ICON = {"ok": "🟢", "warn": "🟠", "info": "🔵"}
flags = diagnostic_flags(df, steps)
cols = st.columns(3)
for i, f in enumerate(flags):
    with cols[i % 3].container(border=True):
        st.markdown(f"{ICON[f['status']]} **{f['name']}**")
        st.markdown(f"### {f['value']}")
        st.caption(f"{f['rule']}")
        with st.expander("What this means"):
            st.write(f["detail"])
st.caption("🟢 fine · 🟠 likely a problem · 🔵 worth a look. Thresholds are heuristics (see `THRESH` in `webapp/analysis.py`).")

# ---- invalid causes -------------------------------------------------------------------------------
st.subheader("Why actions were invalid")
causes = invalid_cause_table(steps)
if causes.empty:
    st.success("No invalid actions in this batch.")
else:
    left, right = st.columns([3, 2])
    with left:
        by_hop = (steps[~steps.valid].assign(label=lambda x: x.cause.map(lambda c: CAUSE_INFO.get(c, (c,))[0]))
                  .groupby(["n_hops", "label"]).size().rename("count").reset_index())
        fig = px.bar(by_hop, x="n_hops", y="count", color="label", labels={"n_hops": "hops to target", "label": "cause"})
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.dataframe(causes[["label", "count", "share_of_invalid", "share_of_actions"]], hide_index=True, width="stretch",
                     column_config={"share_of_invalid": st.column_config.NumberColumn("of invalid", format="percent"),
                                    "share_of_actions": st.column_config.NumberColumn("of all actions", format="percent")})
    with st.expander("What each cause means / most frequent bad answers"):
        for c in causes.cause:
            st.markdown(f"**{CAUSE_INFO.get(c, (c, ''))[0]}**: {CAUSE_INFO.get(c, ('', ''))[1]}")
        st.dataframe(top_bad_titles(steps), hide_index=True, width="stretch")
    if (causes.cause == "unrecorded").any():
        st.caption("Some actions have no stored cause and the run's graph file isn't available locally to infer it.")

# ---- path efficiency ------------------------------------------------------------------------------
st.subheader("Path length vs optimal")
d = df.assign(outcome=df.success.map({True: "success", False: "failure"}))
f = px.strip(d, x="optimal_steps", y="path_len", color="outcome", stripmode="overlay",
             color_discrete_map={"success": "#00CC96", "failure": "#EF553B"},
             labels={"optimal_steps": "optimal steps", "path_len": "steps taken"})
hi = max(d.optimal_steps.max(), d.path_len.max())
f.add_shape(type="line", x0=0, y0=0, x1=hi, y1=hi, line={"dash": "dot", "color": "#9AA0A6"})
st.plotly_chart(f, width="stretch")
st.caption("Dotted line = optimal. Points above it are detours; failures usually pile up at the step limit.")

# ---- reward --------------------------------------------------------------------------------------
st.subheader("Reward")
rb = reward_breakdown(episodes, run.config.get("reward"))
left, right = st.columns(2)
with left:
    st.markdown("**Where reward comes from (mean per episode)**")
    comp = [c for c in ["success bonus", "step cost", "invalid penalty", "dead-end penalty", "progress shaping"] if c in rb.columns]
    m = rb.assign(outcome=rb.end_reason).groupby("outcome")[comp].mean().reset_index().melt(
        id_vars="outcome", var_name="component", value_name="mean reward")
    st.plotly_chart(px.bar(m, x="outcome", y="mean reward", color="component", barmode="relative"), width="stretch")
with right:
    st.markdown("**Total reward distribution**")
    st.plotly_chart(px.histogram(d, x="total_reward", color="outcome", nbins=30,
                                 color_discrete_map={"success": "#00CC96", "failure": "#EF553B"}), width="stretch")
worst = rb.residual.abs().max()
if worst > 1e-6:
    st.warning(f"Reward components don't add up to the logged total (max residual {worst:.3f}): "
               "this run probably used a different reward config than the one recorded in its W&B config.")
else:
    st.caption("✓ Reward components reproduce each episode's logged total exactly.")

# ---- reasoning length ----------------------------------------------------------------------------
if steps.think_chars.gt(0).any():
    st.subheader("Reasoning length")
    st.plotly_chart(px.histogram(steps[steps.think_chars > 0], x="think_chars", nbins=40,
                                 labels={"think_chars": "characters inside <think>"}), width="stretch")
