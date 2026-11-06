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
