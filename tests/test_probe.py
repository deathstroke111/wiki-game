"""Probe helpers: the replayed states and the scoring of one answer."""
import pytest

from wikigame import WikiGraph
from wikigame.probe import collect_states, sample_states, score_answer


@pytest.fixture
def graph():
    nodes = {t: t for t in "ABCDT"}
    return WikiGraph.from_edges(nodes, {"A": ["B", "C", "D"], "B": ["T", "A"], "C": ["A", "B"], "D": [], "T": []})


def answer(title):
    return f'<think>x</think><answer>{{"title": "{title}"}}</answer>'


def test_collect_states_counts_each_state_once_and_tracks_the_path(graph):
    steps = [{"chosen": "Z", "valid": False}, {"chosen": "B", "valid": True}, {"chosen": "T", "valid": True}]
    states = collect_states(graph, [{"target": "T", "path": ["A", "B", "T"], "steps": steps}])
    assert [(s["path"], s["dist"]) for s in states] == [(["A"], 2), (["A", "B"], 1)]


def test_sample_states_is_reproducible_and_pools_far_states():
    states = [{"dist": d, "path": [str(i)], "target": "T"} for i, d in enumerate([2, 2, 3, 3, 5])]
    assert sample_states(states, [2, 3], 3, seed=1) == sample_states(states, [2, 3], 3, seed=1)
    assert sum(s["dist"] == 5 for s in sample_states(states, [2, 3], 3)) == 1


@pytest.mark.parametrize("text,outcome,cause", [
    (answer("B"), "closer", None), (answer("C"), "sideways", None), (answer("D"), "farther", None),
    (answer("Nope"), "invalid", "unknown_title"), ("garbled", "invalid", "unparseable"), (answer("A"), "invalid", "current_page"),
])
def test_score_answer(graph, text, outcome, cause):
    result = score_answer(graph, {"target": "T", "path": ["A"], "dist": 2}, text)
    assert (result["outcome"], result["cause"]) == (outcome, cause)
