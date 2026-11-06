"""Render each Streamlit page headlessly against the synthetic fixture source (no W&B, no network)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app"


@pytest.fixture(autouse=True)
def fixture_source(monkeypatch):
    monkeypatch.setenv("WIKIGAME_APP_SOURCE", "fixture")


def run(page: str) -> AppTest:
    return AppTest.from_file(str(APP / "views" / page), default_timeout=60).run()


def test_runs_page_renders():
    at = run("runs.py")
    assert not at.exception, at.exception
    assert at.title[0].value.endswith("Runs")
    assert len(at.dataframe) >= 1  # runs table
    labels = [m.label for m in at.multiselect]
    assert "Runs to compare" in labels


def test_runs_page_machine_filter():
    at = run("runs.py")
    at.sidebar.multiselect[0].set_value(["cuda"]).run()  # machine filter
    assert not at.exception, at.exception
    assert "1 of 2 runs" in at.caption[0].value


def test_episodes_page_renders_detail_and_graph():
    at = run("episodes.py")
    assert not at.exception, at.exception
    assert at.title[0].value.endswith("Episodes")
    assert len(at.metric) == 5
    assert any("Agent path" in m.value for m in at.markdown)


def test_episodes_page_filters_to_empty():
    at = run("episodes.py")
    at.sidebar.multiselect[0].set_value([]).run()  # no hop selected
    assert not at.exception, at.exception
    assert any("No episodes match" in i.value for i in at.info)
