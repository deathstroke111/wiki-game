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


# ---- invalid causes + feedback ---------------------------------------------------------------------------
def test_invalid_causes(graph):
    env = WikiGameEnv(graph, max_invalid=10)
    env.reset(options={"source": 0, "target": 3})  # A -> ... -> D ; D is not linked from A
    causes = {a: env.step(a)[4]["cause"] for a in [None, "Nope", "A", "D", "C"]}
    assert causes == {None: "unparseable", "Nope": "unknown_title", "A": "current_page",
                      "D": "target_not_linked", "C": "not_linked"}
    env.step("B")
    assert env.step("A")[4]["cause"] == "revisit"


def test_feedback_only_after_invalid_and_only_when_enabled(graph):
    def obs_after(feedback, actions):
        env = WikiGameEnv(graph, seed=0, feedback=feedback)
        out, _ = env.reset(options={"source": 0, "target": 3})
        for a in actions:
            out = env.step(a)[0]
        return out

    msg = "Your last answer was invalid"
    assert msg not in obs_after(True, [])  # nothing wrong yet
    after = obs_after(True, ["D"])
    assert msg in after and "'D' is the target" in after and "2 invalid answers left" in after
    assert msg not in obs_after(False, ["D"])  # flag off
    assert msg not in obs_after(True, ["D", "B"])  # cleared by a valid move


def test_flag_off_observation_unchanged(graph):
    env = WikiGameEnv(graph, seed=0)
    obs, _ = env.reset(options={"source": 0, "target": 3})
    assert "Steps left: 10\n\nLinks:\n" in obs
    assert "Steps left: 10\n\nLinks:\n" in env.step("D")[0]


# ---- shaped reward (off by default) ----------------------------------------------------------------------
def _shaped_env(graph, **kw):
    from wikigame.env import RewardConfig
    return WikiGameEnv(graph, max_invalid=10, reward=RewardConfig(shaping=1.0, **kw), seed=0)


def test_shaping_off_by_default_reproduces_old_rewards(graph):
    env = WikiGameEnv(graph)
    env.reset(options={"source": 0, "target": 3})
    _, r, _, _, info = env.step("B")
    assert abs(r + 0.02) < 1e-9 and "shaping_reward" not in info
    assert abs(env.step("Nope")[1] - (-0.02 - 0.1)) < 1e-9


def test_shaped_reward_ordering_closer_sideways_farther_invalid(graph):
    # path graph A->B->C->D plus A->X dead end; target D. From A: B is closer, X can't reach D (farther).
    env = _shaped_env(graph)
    env.reset(options={"source": 0, "target": 3})
    r_closer = env.step("B")[1]
    env.reset(options={"source": 0, "target": 3})
    r_farther = env.step("X")[1]  # unreachable from X: treated as -1; also a dead end (-0.2)
    env.reset(options={"source": 0, "target": 3})
    r_invalid = env.step("Nope")[1]
    assert r_closer > 0.9 and r_farther < 0 and r_invalid < r_farther < r_closer
    assert r_invalid <= -2.0


def test_shaping_is_clipped_and_recorded(graph):
    from wikigame.rollout import run_episode
    env = _shaped_env(graph)
    ep = run_episode(env, lambda o, e: '<answer>{"title": "B"}</answer>' if e.cur == 0 else
                     ('<answer>{"title": "C"}</answer>' if e.cur == 1 else '<answer>{"title": "D"}</answer>'),
                     options={"source": 0, "target": 3})
    assert ep.success and [s["shaping_reward"] for s in ep.steps] == [1.0, 1.0, 1.0]
    assert abs(ep.total_reward - (3 * 1.0 + 1.0 - 3 * 0.02)) < 1e-9
