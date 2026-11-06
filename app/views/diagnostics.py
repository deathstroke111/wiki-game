"""Diagnostics: format/validity health, path efficiency, reward decomposition, reward-hacking flags."""
import plotly.express as px
import streamlit as st

from wikigame.webapp import st_helpers as H
from wikigame.webapp.analysis import diagnostic_flags, episodes_df, reward_breakdown, steps_df

st.title("🩺 Behaviour diagnostics")
H.refresh_button()
run, eset, episodes, graph = H.pick_episode_set()
df, steps = episodes_df(episodes), steps_df(episodes)
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
    comp = ["success bonus", "step cost", "invalid penalty", "dead-end penalty"]
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
