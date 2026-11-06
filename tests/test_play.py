import pytest

from wikigame import WikiGraph
from wikigame.eval import make_pairs
from wikigame.rollout import oracle_policy
from wikigame.webapp.play import human_text, new_game, pair_for, upload_human_run
from wikigame.env import parse_action


@pytest.fixture
def graph():
    nodes = {t: f"About {t}" for t in "ABCDEFGH"}
    edges = {"A": ["B", "C"], "B": ["D", "A"], "C": ["E", "F"], "D": ["G"], "E": ["G", "H"],
             "F": ["H"], "G": ["H"], "H": ["A"]}
    return WikiGraph.from_edges(nodes, edges)


def test_human_text_round_trips():
    assert parse_action(human_text('Odd "quoted" title')) == 'Odd "quoted" title'


def test_pair_for_held_out_matches_eval_pairs(graph):
    pairs = make_pairs(graph, hops=(2,), n_per_hop=50)[2]
    assert [pair_for(graph, 2, i, True, 0) for i in range(3)] == pairs[:3]
    assert pair_for(graph, 2, 50, True, 0) == pairs[0]  # wraps
    s, t = pair_for(graph, 3, 0, False, seed=7)
    assert graph.shortest_dist(s, t) == 3


def test_human_game_flow_and_upload_offline(graph, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WIKIGAME_MACHINE", "local")
    env, rec, obs = new_game(graph, 2, 8, pair_for(graph, 2, 0, True, 0))
    assert "Target:" in obs
    pol, done = oracle_policy(), False
    while not done:  # play the optimal route "by hand"
        title = parse_action(pol(obs, env))
        obs, r, term, trunc, info = env.step(title)
        rec.record(human_text(title), title, r, info)
        done = term or trunc
    ep = rec.finish()
    assert ep.success and ep.extra_steps == 0 and ep.n_hops == 2
    url = upload_human_run([ep], graph, "tiny", 8)
    assert url is None or isinstance(url, str)  # offline runs have no URL; must not raise
