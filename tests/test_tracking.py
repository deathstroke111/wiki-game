import pytest

from wikigame import WikiGameEnv, WikiGraph
from wikigame import tracking
from wikigame.eval import evaluate, make_pairs
from wikigame.rollout import oracle_policy


@pytest.fixture
def graph():
    nodes = {t: f"About {t}" for t in "ABCD"}
    return WikiGraph.from_edges(nodes, {"A": ["B"], "B": ["C"], "C": ["D"], "D": ["A"]})


def test_detect_machine_override(monkeypatch):
    monkeypatch.setenv("WIKIGAME_MACHINE", "cuda")
    assert tracking.detect_machine() == "cuda"
    monkeypatch.setenv("WIKIGAME_MACHINE", "tpu")
    with pytest.raises(ValueError):
        tracking.detect_machine()


def test_run_identity_and_logging_offline(monkeypatch, tmp_path, graph):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WIKIGAME_MACHINE", "local")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    run = tracking.init_run("unit", "smoke", {"x": 1}, graph, "tiny", tags=["t"])
    try:
        assert run.entity == "aiwizards" and run.project == "wiki-game"
        assert run.group == "unit" and run.job_type == "smoke"
        assert run.name.startswith("unit-local-")
        assert {"machine:local", "graph:tiny", "t"} <= set(run.tags)
        assert run.config["machine"] == "local" and run.config["graph_nodes"] == 4 and run.config["x"] == 1

        env = WikiGameEnv(graph, max_steps=5, seed=0)
        metrics, episodes = evaluate(env, oracle_policy(), make_pairs(graph, hops=(1, 2), n_per_hop=3))
        tracking.log_eval(run, metrics)
        tracking.log_episodes(run, episodes, name="eval")
        assert run.summary["eval/success"] == 1.0
    finally:
        run.finish()
