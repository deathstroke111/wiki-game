"""Cached Streamlit accessors over the DataSource. Short TTLs give near-live refresh while training."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..graph import WikiGraph
from .data import DataSource, EpisodeSet, RunInfo, get_source

TTL = 30  # seconds


@st.cache_resource
def source() -> DataSource:
    return get_source()


@st.cache_data(ttl=TTL, show_spinner="Loading runs from W&B…")
def runs() -> list[RunInfo]:
    return source().list_runs()


@st.cache_data(ttl=TTL, show_spinner="Loading run history…")
def history(run_id: str) -> pd.DataFrame:
    return source().run_history(run_id)


@st.cache_data(ttl=TTL)
def episode_sets(run_id: str) -> list[EpisodeSet]:
    return source().episode_sets(run_id)


@st.cache_data(ttl=None, show_spinner="Loading episodes…")  # artifacts are immutable per key
def episodes(run_id: str, key: str) -> list[dict]:
    return source().load_episodes(run_id, key)


@st.cache_resource
def graph(name: str | None) -> WikiGraph | None:
    return source().load_graph(name)


def refresh_button() -> None:
    if st.sidebar.button("🔄 Refresh from W&B", width="stretch"):
        st.cache_data.clear()
        st.rerun()


def available_graphs() -> list[str]:
    """Graph files the Play page can use: data/*.json graphs (fixture mode: a synthetic graph)."""
    import os

    from .data import GRAPH_DIR

    if os.environ.get("WIKIGAME_APP_SOURCE") == "fixture":
        return ["fixture"]
    return sorted(p.stem for p in GRAPH_DIR.glob("*.json"))


def pick_episode_set(allow_missing_graph: bool = True):
    """Sidebar run + eval-batch pickers shared by the episode-based pages.

    Selection persists across pages (session_state) and can be preset by ?run=<id>&set=<key> in the URL.
    Returns (run, episode_set, episodes, graph). Calls st.stop() when there is nothing to show.
    """
    all_runs = runs()
    if not all_runs:
        st.info("No runs found in the W&B project yet.")
        st.stop()
    qp = st.query_params
    want_run = qp.get("run") or st.session_state.get("_run_id")
    labels = {f"{r.name}  ·  {r.id}": r for r in all_runs}
    keys = list(labels)
    idx = next((i for i, k in enumerate(keys) if labels[k].id == want_run), 0)
    run = labels[st.sidebar.selectbox("Run", keys, index=idx)]
    st.session_state["_run_id"] = run.id

    sets = episode_sets(run.id)
    if not sets:
        st.warning("This run has no `episodes` artifact (it never logged episodes).")
        st.stop()
    set_labels = {f"{s.split} · step {s.step} · {s.n_episodes} episodes": s for s in sets}
    skeys = list(set_labels)
    want_set = qp.get("set") or st.session_state.get("_set_key")
    sidx = next((i for i, k in enumerate(skeys) if set_labels[k].key == want_set), len(skeys) - 1)
    eset = set_labels[st.sidebar.selectbox("Eval batch", skeys, index=sidx)]
    st.session_state["_set_key"] = eset.key
    return run, eset, episodes(run.id, eset.key), graph(run.graph)
