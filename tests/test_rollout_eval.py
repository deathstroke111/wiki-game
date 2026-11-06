import pytest

from wikigame import WikiGameEnv, WikiGraph
from wikigame.eval import evaluate, make_pairs
from wikigame.rollout import oracle_policy, random_policy, run_episode


@pytest.fixture
def graph():
    # layered DAG-ish graph with branching and a back-edge so random walks can fail
    nodes = {t: f"About {t}" for t in "ABCDEFGH"}
    edges = {
        "A": ["B", "C"], "B": ["D", "A"], "C": ["E", "F"], "D": ["G"], "E": ["G", "H"],
        "F": ["H"], "G": ["H"], "H": ["A"],
    }
    return WikiGraph.from_edges(nodes, edges)


def test_oracle_is_optimal(graph):
    env = WikiGameEnv(graph, max_steps=10, max_links_shown=500, seed=0)
    metrics, eps = evaluate(env, oracle_policy(), make_pairs(graph, hops=(1, 2, 3), n_per_hop=20))
    assert metrics["success"] == 1.0 and metrics["invalid_rate"] == 0.0
    assert all(e.extra_steps == 0 and e.end_reason == "success" for e in eps)
    assert metrics["extra_steps@2"] == 0


def test_random_is_worse_than_oracle(graph):
    pairs = make_pairs(graph, hops=(3,), n_per_hop=100)
    env = WikiGameEnv(graph, max_steps=10, max_links_shown=500, seed=0)
    rnd, _ = evaluate(env, random_policy(0), pairs)
    orc, _ = evaluate(env, oracle_policy(), pairs)
    assert rnd["success@3"] < orc["success@3"]
    assert rnd["reward"] < orc["reward"]


def test_make_pairs_deterministic_and_exact(graph):
    a, b = make_pairs(graph, n_per_hop=5, hops=(2,)), make_pairs(graph, n_per_hop=5, hops=(2,))
    assert a == b
    assert all(graph.shortest_dist(s, t) == 2 for s, t in a[2])


def test_garbage_policy_records_invalid_and_truncates(graph):
    env = WikiGameEnv(graph, max_invalid=3, seed=0)
    ep = run_episode(env, lambda obs, env: "no answer tags", options={"source": 0, "target": 7})
    assert not ep.success and ep.n_invalid == 3 and ep.n_valid == 0
    assert ep.end_reason == "invalid_action" and ep.path == ["A"]


def test_episode_serializable(graph):
    env = WikiGameEnv(graph, seed=0)
    ep = run_episode(env, oracle_policy(), options={"source": 0, "target": 6}, n_hops=3)
    d = ep.to_dict()
    assert d["path"][0] == "A" and d["path"][-1] == "G" and d["n_hops"] == 3
