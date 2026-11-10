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


# ---- Batch B pages ------------------------------------------------------------------------------------
def test_failures_page_renders():
    at = run("failures.py")
    assert not at.exception, at.exception
    assert at.title[0].value.endswith("Failure breakdown")
    assert len(at.metric) == 3  # wrong-turn rates (all / failed / successful)
    assert any("Worst failures" in s.value for s in at.subheader)


def test_failures_page_on_oracle_run_has_no_failures():
    at = run("failures.py")
    runs = at.sidebar.selectbox[0].options
    at.sidebar.selectbox[0].select([o for o in runs if "oracle" in o][0]).run()
    assert not at.exception, at.exception
    assert any("No failed episodes" in s.value for s in at.success)


def test_diagnostics_page_renders_flags():
    at = run("diagnostics.py")
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Invalid actions" in text and "Format compliance" in text
    assert any("reproduce each episode" in c.value for c in at.caption)  # reward components add up
    assert not at.warning


def test_play_page_full_game_and_give_up():
    at = run("play.py")
    assert not at.exception, at.exception
    assert any("New game" in i.value for i in at.info)
    at.sidebar.slider[0].set_value(1)
    at.sidebar.button[0].click().run()  # ▶ New game
    assert not at.exception, at.exception
    assert any("Target:" in m.value for m in at.markdown)

    for _ in range(10):  # keep clicking the first link button until the game ends
        if at.session_state["pl_done"] is not None:
            break
        link_buttons = [b for b in at.button if b.key and b.key.startswith("lnk_")]
        link_buttons[0].click().run()
        assert not at.exception, at.exception
    done = at.session_state["pl_done"]
    assert done is not None and len(at.session_state["pl_games"]) == 1
    assert any("Your games (1)" in s.value for s in at.subheader)

    at.sidebar.button[0].click().run()  # second game, give up straight away
    give_up = [b for b in at.button if b.label.endswith("Give up")][0]
    give_up.click().run()
    assert not at.exception, at.exception
    assert at.session_state["pl_games"][-1].end_reason == "gave_up"


# ---- Batch C page -------------------------------------------------------------------------------------
def test_baselines_page_renders_and_filters():
    at = run("baselines.py")
    assert not at.exception, at.exception
    assert at.title[0].value.endswith("Baseline comparison")
    assert any("No human baseline" in c.value for c in at.caption)  # nudges toward the Play page
    at.multiselect[0].set_value([at.multiselect[0].options[0]]).run()  # single agent
    assert not at.exception, at.exception


# ---- statistics page ----------------------------------------------------------------------------------
def test_statistics_page_sections_render():
    at = run("statistics.py")
    assert not at.exception, at.exception
    assert at.title[0].value.endswith("Statistics")
    assert any("Compare any two" in s.value for s in at.subheader)
    at.button_group[0].set_value("Failure anatomy").run()
    assert not at.exception, at.exception
    assert any("What the agent does" in s.value for s in at.subheader)
    at.button_group[0].set_value("Training process").run()
    assert not at.exception, at.exception
    assert any("No training runs" in i.value for i in at.info)


def test_statistics_page_with_no_agents_picked():
    at = run("statistics.py")
    at.sidebar.multiselect[0].set_value([]).run()
    assert not at.exception, at.exception
    assert any("Pick at least one" in i.value for i in at.info)


# ---- invalid-cause views ------------------------------------------------------------------------------
@pytest.fixture
def source_with_invalids(monkeypatch):
    """Fixture source plus a run whose agent only emits invalid actions (the stock runs never do)."""
    import streamlit as st

    from wikigame import WikiGameEnv
    from wikigame.webapp import data
    from wikigame.rollout import run_episode

    real = data.make_fixture_source

    def with_garbage():
        src = real()
        eps = []
        for k in (1, 2):
            for s, t in [(0, 3), (0, 7)]:
                ep = run_episode(WikiGameEnv(src.graph, max_invalid=3, seed=0), lambda o, e: "no tags",
                                 options={"source": s, "target": t}, n_hops=k)
                eps.append(ep.to_dict())
        key = "episodes-run9-eval-step0:v0"
        src.runs.append(data.RunInfo("run9", "zeroshot-garbage", "local", "g", "baseline", "finished",
                                     ["machine:local"], "2026-11-09", {"graph": "fixture"}, {}))
        src.sets["run9"] = [data.EpisodeSet(key, 0, "eval", len(eps), "d9")]
        src.episodes[("run9", key)] = eps
        return src

    st.cache_resource.clear(); st.cache_data.clear()
    monkeypatch.setattr(data, "make_fixture_source", with_garbage)
    yield
    st.cache_resource.clear(); st.cache_data.clear()


def _select_run(at, name):
    box = at.sidebar.selectbox[0]
    box.select([o for o in box.options if name in o][0]).run()


def test_diagnostics_shows_invalid_causes(source_with_invalids):
    at = run("diagnostics.py")
    _select_run(at, "zeroshot-garbage")
    assert not at.exception, at.exception
    assert any("Why actions were invalid" in s.value for s in at.subheader)
    tables = [d.value for d in at.dataframe if "label" in d.value.columns]
    assert tables and list(tables[0]["label"]) == ["Unparseable output"]  # every garbage action is unparseable


def test_episodes_page_shows_why_invalid_column(source_with_invalids):
    at = run("episodes.py")
    _select_run(at, "zeroshot-garbage")
    assert not at.exception, at.exception
    steps_tables = [d.value for d in at.dataframe if "why invalid" in d.value.columns]
    assert steps_tables and (steps_tables[0]["why invalid"] == "Unparseable output").all()
