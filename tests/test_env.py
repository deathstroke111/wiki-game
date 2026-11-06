import random

import pytest

from wikigame import WikiGameEnv, WikiGraph, parse_action


@pytest.fixture
def graph():
    # A -> B -> C -> D ; A -> X (dead end) ; B -> A (cycle back)
    nodes = {t: f"About {t}" for t in "ABCDX"}
    edges = {"A": ["B", "X"], "B": ["C", "A"], "C": ["D"], "D": [], "X": []}
    return WikiGraph.from_edges(nodes, edges)


def play(env, titles, target=None):
    env.reset(options={"source": env.graph.id_of(titles[0]), "target": env.graph.id_of(target or titles[-1])})
    out = None
    for t in titles[1:]:
        out = env.step(t)
    return out


def test_bfs_and_pairs(graph):
    assert graph.shortest_dist(0, 3) == 3
    s, t = graph.sample_pair(2, random.Random(0))
    assert graph.shortest_dist(s, t) == 2


def test_optimal_path_succeeds(graph):
    _, r, term, trunc, info = play(WikiGameEnv(graph, max_steps=10), list("ABCD"))
    assert term and not trunc and info["success"] and info["extra_steps"] == 0 and r > 0.5


def test_dead_end_terminates(graph):
    _, r, term, _, info = play(WikiGameEnv(graph), ["A", "X"], target="D")
    assert term and info["reason"] == "dead_end" and r < 0


def test_revisit_and_unknown_are_invalid(graph):
    env = WikiGameEnv(graph, max_invalid=2)
    env.reset(options={"source": 0, "target": 3})
    env.step("B")
    _, _, term, trunc, info = env.step("A")  # revisit
    assert not info["valid"] and not term and not trunc
    _, _, _, trunc, _ = env.step("Nonexistent")
    assert trunc  # max_invalid reached


def test_step_limit_truncates(graph):
    env = WikiGameEnv(graph, max_steps=1)
    _, _, term, trunc, info = play(env, ["A", "B"], target="D")
    assert trunc and info["reason"] == "step_limit"


def test_parse_action():
    assert parse_action('<think>hm</think><answer>{"title": "Foo"}</answer>') == "Foo"
    assert parse_action("no tags") is None
    assert parse_action("<answer>not json</answer>") is None
    assert parse_action('<answer>["Foo"]</answer>') is None
