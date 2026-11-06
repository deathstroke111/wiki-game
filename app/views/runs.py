"""Runs: compare runs (Mac vs CUDA, baselines vs trained) on eval metrics and training curves."""
import pandas as pd
import plotly.express as px
import streamlit as st

from wikigame.webapp import st_helpers as H
from wikigame.webapp.analysis import hop_metric_df, runs_df

st.title("📊 Runs")
H.refresh_button()

all_runs = H.runs()
if not all_runs:
    st.info("No runs found in the W&B project yet. Run `uv run python scripts/smoke_run.py` first.")
    st.stop()

# ---- filters --------------------------------------------------------------------------------------
df_all = runs_df(all_runs)
with st.sidebar:
    machines = st.multiselect("Machine", sorted(df_all.machine.unique()), default=sorted(df_all.machine.unique()))
    job_types = st.multiselect("Job type", sorted(df_all.job_type.unique()), default=sorted(df_all.job_type.unique()))
    groups = st.multiselect("Experiment group", sorted(df_all.group.unique()))
mask = df_all.machine.isin(machines) & df_all.job_type.isin(job_types)
if groups:
    mask &= df_all.group.isin(groups)
df = df_all[mask]
st.caption(f"{len(df)} of {len(df_all)} runs · project `aiwizards/wiki-game`")

st.dataframe(
    df.drop(columns=["id"]), hide_index=True, width="stretch",
    column_config={"success": st.column_config.ProgressColumn("success", min_value=0, max_value=1, format="%.2f"),
                   "path_len": st.column_config.NumberColumn("path len", format="%.2f"),
                   "invalid_rate": st.column_config.NumberColumn("invalid rate", format="%.3f"),
                   "reward": st.column_config.NumberColumn(format="%.3f")},
)

# ---- selection ------------------------------------------------------------------------------------
by_label = {f"{r.run}  ·  {r.id}": r.id for r in df.itertuples()}
picked = st.multiselect("Runs to compare", list(by_label), default=list(by_label)[:4])
sel_ids = [by_label[p] for p in picked]
sel_runs = [r for r in all_runs if r.id in sel_ids]
if not sel_runs:
    st.stop()

tab_eval, tab_curves = st.tabs(["Eval by hop distance", "Training curves"])

with tab_eval:
    c1, c2 = st.columns(2)
    for col, (name, title, fmt) in zip((c1, c2), (("success", "Success rate @ hops", ".0%"),
                                                  ("extra_steps", "Extra steps above optimal @ hops", ".2f"))):
        d = hop_metric_df(sel_runs, name)
        with col:
            st.subheader(title)
            if d.empty:
                st.caption(f"No `eval/{name}@k` metrics in the selected runs"
                           + (" (extra steps are only defined where the agent succeeded)." if name == "extra_steps" else "."))
                continue
            d["hop"] = d.hop.astype(str)
            fig = px.bar(d, x="hop", y="value", color="run", barmode="group", text_auto=fmt,
                         labels={"hop": "hops to target", "value": name.replace("_", " ")})
            fig.update_layout(legend={"orientation": "h", "y": -0.25}, margin={"t": 10})
            if name == "success":
                fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
            st.plotly_chart(fig, width="stretch")


def curves() -> None:
    frames = []
    for r in sel_runs:
        h = H.history(r.id)
        if not h.empty and "_step" in h:
            h = h.assign(run=r.name, machine=r.machine)
            frames.append(h)
    if not frames:
        st.info("No step-wise history yet. Training runs log `train/*` metrics here; baseline runs log a single eval.")
        return
    hist = pd.concat(frames, ignore_index=True)
    metrics = sorted(c for c in hist.columns if c not in ("_step", "_runtime", "_timestamp", "run", "machine"))
    default = [m for m in ("train/reward_mean", "train/success_rate") if m in metrics] or metrics[:1]
    chosen = st.multiselect("Metrics", metrics, default=default)
    cols = st.columns(2)
    for i, m in enumerate(chosen):
        d = hist.dropna(subset=[m])
        fig = px.line(d, x="_step", y=m, color="run", line_dash="machine", markers=len(d) < 40 * len(sel_runs),
                      color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(margin={"t": 30}, title=m, legend={"orientation": "h", "y": -0.3})
        cols[i % 2].plotly_chart(fig, width="stretch")


with tab_curves:
    live = st.toggle("Auto-refresh every 30 s (live training)", value=False,
                     help="History is cached for 30 s, so each tick fetches fresh points from W&B.")
    st.fragment(run_every=30 if live else None)(curves)()

