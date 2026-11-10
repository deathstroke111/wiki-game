"""Statistics helpers checked against hand-computed values and tiny constructed games."""
import numpy as np
import pandas as pd
import pytest

from wikigame import WikiGraph
from wikigame.webapp import stats


def ep(hops, source, target, success):
    return {"n_hops": hops, "source": source, "target": target, "success": success}


@pytest.mark.parametrize("a,b,expected", [(16, 6, 0.0525), (27, 13, 0.0385), (0, 5, 0.0625), (0, 0, 1.0)])
def test_mcnemar_exact(a, b, expected):
    assert stats.mcnemar_exact(a, b) == pytest.approx(expected, abs=5e-4)


def test_holm_known_example():
    np.testing.assert_allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert stats.holm([0.9, 0.9]).max() == 1.0
    assert stats.holm([]).size == 0


def test_bootstrap_ci_collapses_when_differences_are_constant():
    assert stats.bootstrap_diff_ci(np.ones(20), np.array([1, 2] * 10)) == (1.0, 1.0)


def test_bootstrap_ci_contains_the_estimate():
    d = np.array([1.0] * 12 + [-1.0] * 4 + [0.0] * 24)
    low, high = stats.bootstrap_diff_ci(d, np.array([1, 2] * 20))
    assert low < d.mean() < high


def test_success_matrix_aligns_games_and_keeps_first_attempt():
    m = stats.success_matrix({"x": [ep(1, "A", "B", True), ep(2, "A", "C", False)], "y": [ep(1, "A", "B", False)]})
    assert m.x.tolist() == [1.0, 0.0] and m.y.iloc[0] == 0.0 and pd.isna(m.y.iloc[1])
    repeated = stats.success_matrix({"x": [ep(1, "A", "B", True), ep(1, "A", "B", False)]})
    assert repeated.x.tolist() == [1.0]


def test_compare_only_uses_shared_games():
    x = [ep(1, "A", str(i), True) for i in range(6)]
    y = [ep(1, "A", str(i), i == 0) for i in range(4)]
    wide = stats.success_matrix({"x": x, "y": y})
    row = stats.compare(wide, "x", "y")
    assert (row["games"], row["a_only"], row["b_only"]) == (4, 3, 0)
    assert row["diff"] == pytest.approx(0.75)


def test_compare_empty_hop_is_not_an_error():
    wide = stats.success_matrix({"x": [ep(1, "A", "B", True)], "y": [ep(1, "A", "B", False)]})
    assert stats.compare(wide, "x", "y", hop=4)["games"] == 0


def test_planned_pairs_follow_training_order_regardless_of_input_order():
    a0, s1, s2a, s2, s3 = ("zeroshot-r1-1.5b-lenient", "grpo-step-r1-stage1b-final", "grpo-step-r1-stage2-step100",
                           "grpo-step-r1-stage2-final", "grpo-step-r1-stage3-final")
    q = "zeroshot-qwen2.5-1.5b-instruct-lenient"
    assert stats.planned_pairs([s3, q, s2, a0, s2a, s1, "zeroshot-qwen2.5-1.5b-instruct"]) == [
        (s1, a0), (s2a, s1), (s2, s2a), (s3, s2), (s3, a0), (s3, q)]
    assert stats.planned_pairs(["random", "oracle"]) == []


def test_conditional_logit_recovers_direction():
    # 'b' beats 'a' in 15 split games and loses 5, so the conditional odds ratio is exactly 15 / 5.
    a = [ep(1, "A", str(i), i >= 15) for i in range(20)] + [ep(1, "A", f"t{i}", i % 2 == 0) for i in range(10)]
    b = [ep(1, "A", str(i), i < 15) for i in range(20)] + [ep(1, "A", f"t{i}", i % 2 == 0) for i in range(10)]
    t = stats.conditional_logit(stats.success_matrix({"a": a, "b": b}), ["a", "b"], "a")
    assert t.loc[0, "agent"] == "b" and t.loc[0, "odds_ratio"] == pytest.approx(3.0, rel=1e-3)
    assert t.loc[0, "or_low"] < 3 < t.loc[0, "or_high"]


def test_games_needed_grows_for_smaller_effects():
    assert stats.games_needed(0.05, 0.28) > stats.games_needed(0.15, 0.28) > 0


def test_training_change_uses_one_value_per_step():
    hist = pd.DataFrame({stats.STEP_COL: list(range(1, 101)) * 2, "m": [0.0] * 50 + [1.0] * 50 + [None] * 100})
    t = stats.first_last_change(hist, {"m": "metric"}, window=50)
    assert t.loc[0, "change"] == pytest.approx(1.0)
    b = stats.block_means(hist, "m", block=50)
    assert b.step.tolist() == [50, 100] and b.value.tolist() == [0.0, 1.0]


# A -> B -> T is the shortest route. C is a side page at the same distance as A's start, D leads nowhere.
@pytest.fixture
def graph():
    nodes = {t: t for t in "ABCDT"}
    return WikiGraph.from_edges(nodes, {"A": ["B", "C", "D"], "B": ["T", "A"], "C": ["A", "B"], "D": [], "T": []})


def game(path, steps, success, end="step_limit"):
    return {"n_hops": 2, "source": path[0], "target": "T", "path": path, "steps": steps, "success": success, "end_reason": end,
            "n_valid": sum(s["valid"] for s in steps), "n_invalid": sum(not s["valid"] for s in steps)}


def move(title):
    return {"chosen": title, "valid": True, "reward": 0.0, "n_links": 3, "text": ""}


def bad(title, cause="not_linked"):
    return {"chosen": title, "valid": False, "reward": 0.0, "n_links": 3, "text": "", "cause": cause}


def test_action_table_classifies_each_answer(graph):
    eps = [game(["A", "B", "T"], [bad("Z"), bad("Z"), move("B"), move("T")], True, "success"),
           game(["A", "D"], [move("D")], False, "dead_end")]
    a = stats.action_table(graph, eps)
    assert a.outcome.tolist() == ["invalid", "invalid", "closer", "closer", "farther"]
    assert a.dist_before.tolist() == [2, 2, 2, 1, 2]
    assert a.repeat_invalid.tolist() == [False, True, False, False, False]
    # from A, only B of {B, C, D} gets closer
    assert a.chance_closer[0] == pytest.approx(1 / 3)


def test_visited_pages_leave_the_chance_denominator(graph):
    a = stats.action_table(graph, [game(["A", "B", "T"], [move("B"), move("T")], True, "success")])
    assert a.chance_closer.tolist() == pytest.approx([1 / 3, 1.0])  # from B, A is already visited


def test_sideways_move_is_not_closer(graph):
    a = stats.action_table(graph, [game(["A", "C"], [move("C")], False)])
    assert a.outcome.tolist() == ["sideways"]


def test_outcome_by_distance_shares_sum_to_one(graph):
    eps = [game(["A", "B", "T"], [bad("Z"), move("B"), move("T")], True, "success")]
    t = stats.outcome_by_distance(stats.action_table(graph, eps))
    assert t[stats.OUTCOMES].sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    d2 = t[t.dist == "2"].iloc[0]
    assert d2.answers == 2 and d2.invalid == 0.5 and d2.closer_if_valid == 1.0 and d2.lift == pytest.approx(3.0)


def test_behaviour_summary_of_a_failed_game_one_click_from_target(graph):
    eps = [game(["A", "B"], [move("B"), bad("Q"), bad("Q"), bad("Q")], False, "invalid_action")]
    s = stats.behaviour_summary(stats.action_table(graph, eps), stats.game_table(graph, eps))
    assert s["invalid"] == 0.75 and s["repeat_share_of_invalid"] == pytest.approx(2 / 3)
    assert s["fail_end_invalid"] == 1.0 and s["fail_one_click_away"] == 1.0 and s["fail_got_closer"] == 1.0
