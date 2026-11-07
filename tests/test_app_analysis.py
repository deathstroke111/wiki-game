import pytest

from wikigame import WikiGraph
from wikigame.webapp.analysis import (episodes_df, hop_metric_df, optimal_path, path_subgraph, runs_df,
                                      step_progress)
from wikigame.webapp.data import make_fixture_source
from wikigame.webapp.viz import path_figure


@pytest.fixture(scope="module")
def src():
    return make_fixture_source()


@pytest.fixture
def graph(src):
    return src.graph


def _ep(src, run="run0"):
    key = src.sets[run][0].key
    return src.episodes[(run, key)]


def test_shortest_path_graph():
    g = WikiGraph.from_edges({t: "" for t in "ABCD"}, {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []})
    p = g.shortest_path(0, 3)
    assert p is not None and len(p) == 3 and p[0] == 0 and p[-1] == 3
    assert g.shortest_path(3, 0) is None and g.shortest_path(1, 1) == [1]


def test_episodes_df_and_runs_df(src):
    df = episodes_df(_ep(src))
    assert len(df) == 30 and {"success", "end_reason", "path_len", "label"} <= set(df.columns)
    rdf = runs_df(src.runs)
    assert set(rdf.machine) == {"local", "cuda"} and rdf.set_index("run").loc["baseline-oracle-cuda", "success"] == 1.0


def test_hop_metric_df(src):
    d = hop_metric_df(src.runs, "success")
    assert set(d.hop) == {1, 2, 3} and (d[d.run == "baseline-oracle-cuda"].value == 1.0).all()
    assert hop_metric_df(src.runs, "nonexistent").empty


def test_oracle_matches_optimal_and_has_no_wrong_turns(src, graph):
    for ep in _ep(src, "run1"):  # oracle run
        opt = optimal_path(graph, ep)
        assert len(opt) == len(ep["path"])  # ties: several shortest paths can exist, so compare length
        assert opt[0] == ep["path"][0] and opt[-1] == ep["path"][-1]
        assert not any(r["wrong_turn"] for r in step_progress(graph, ep))


def test_random_agent_has_wrong_turns_and_flags_nodes(src, graph):
    bad = [ep for ep in _ep(src, "run0") if not ep["success"]]
    assert bad, "random agent should fail sometimes on this graph"
    ep = bad[0]
    prog = step_progress(graph, ep)
    assert len(prog) == len(ep["path"]) - 1 and any(r["wrong_turn"] for r in prog)
    sub = path_subgraph(graph, ep, n_neighbors=2, seed=1)
    assert any(n["wrong"] for n in sub["nodes"].values())
    assert sub["nodes"][ep["source"]]["kinds"] >= {"source", "agent"}
    assert sub["nodes"][ep["target"]]["kinds"] >= {"target", "optimal"}


def test_subgraph_is_deterministic_and_edges_reference_nodes(src, graph):
    ep = _ep(src, "run0")[5]
    a, b = path_subgraph(graph, ep, seed=3), path_subgraph(graph, ep, seed=3)
    assert a["edges"] == b["edges"]
    assert all(u in a["nodes"] and v in a["nodes"] for u, v, _ in a["edges"])
    assert {k for _, _, k in a["edges"]} <= {"agent", "optimal", "neighbor"}


def test_path_figure_builds(src, graph):
    fig = path_figure(path_subgraph(graph, _ep(src, "run0")[0]))
    assert len(fig.data) >= 3


# ---- Batch B ----------------------------------------------------------------------------------------
from wikigame.rollout import run_episode  # noqa: E402
from wikigame import WikiGameEnv  # noqa: E402
from wikigame.webapp.analysis import (diagnostic_flags, drift_df, end_reason_table, first_wrong_turn,  # noqa: E402
                                      reward_breakdown, steps_df, wrong_turn_by_step, worst_failures)


def test_drift_oracle_none_random_some(src, graph):
    assert not drift_df(graph, _ep(src, "run1")).wrong_turn.any()
    d = drift_df(graph, _ep(src, "run0"))
    assert d.wrong_turn.any() and len(d) == sum(len(e["path"]) - 1 for e in _ep(src, "run0"))
    w = wrong_turn_by_step(d)
    assert (w.share.between(0, 1)).all() and w.moves.iloc[0] == d[d.step == 1].shape[0]
    fw = first_wrong_turn(d)
    assert (fw >= 1).all()


def test_end_reasons_and_worst_failures(src, graph):
    df = episodes_df(_ep(src, "run0"))
    t = end_reason_table(df)
    assert t.values.sum() == len(df) and set(t.index) == {1, 2, 3}
    bad = worst_failures(df, drift_df(graph, _ep(src, "run0")), k=5)
    assert len(bad) <= 5 and not bad.success.any()
    assert bad.wrong_turn_share.is_monotonic_decreasing


def test_reward_breakdown_reproduces_env_totals(src):
    """The decomposition (success/step/invalid/dead-end) must add up exactly to what the env logged."""
    for rid in ("run0", "run1"):
        rb = reward_breakdown(_ep(src, rid), {"step_cost": 0.02})
        assert rb.residual.abs().max() < 1e-9


def test_reward_breakdown_with_invalid_actions_and_dead_end(graph):
    eps = [run_episode(WikiGameEnv(graph, max_invalid=3, seed=0), lambda o, e: "garbage", options={"source": 0, "target": 7}).to_dict(),
           run_episode(WikiGameEnv(graph, seed=0), lambda o, e: '<answer>{"title": "B"}</answer>' if e.cur == 0 else '<answer>{"title": "D"}</answer>',
                       options={"source": 0, "target": 4}).to_dict()]
    assert eps[0]["end_reason"] == "invalid_action"
    assert reward_breakdown(eps).residual.abs().max() < 1e-9


def test_steps_df_format_compliance(graph):
    good = '<think>ok</think><answer>{"title": "B"}</answer>'
    ep = run_episode(WikiGameEnv(graph, max_invalid=2, seed=0), lambda o, e: good if e.steps == 0 and not e.invalid else "nope",
                     options={"source": 0, "target": 7}).to_dict()
    s = steps_df([ep])
    assert s.format_ok.tolist()[0] is True and not s.format_ok.iloc[1:].any()
    assert s.think_chars.iloc[0] == 2


def test_flags_oracle_clean_and_garbage_flagged(src, graph):
    clean = diagnostic_flags(episodes_df(_ep(src, "run1")), steps_df(_ep(src, "run1")))
    assert all(f["status"] != "warn" for f in clean)
    garbage = [run_episode(WikiGameEnv(graph, max_invalid=3, seed=0), lambda o, e: "x", options={"source": 0, "target": 7}, n_hops=2).to_dict()
               for _ in range(5)]
    warned = {f["name"] for f in diagnostic_flags(episodes_df(garbage), steps_df(garbage)) if f["status"] == "warn"}
    assert {"Invalid actions", "Format compliance", "Quitting via invalid actions"} <= warned


# ---- Batch C ----------------------------------------------------------------------------------------
from wikigame.webapp.analysis import (baseline_table, comparability_warnings, has_hop_results,  # noqa: E402
                                      hop_n, latest_per_policy, policy_label, wilson_interval)


def test_wilson_interval_properties():
    lo, hi = wilson_interval(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(1, 2)[1] - wilson_interval(1, 2)[0] > wilson_interval(50, 100)[1] - wilson_interval(50, 100)[0]
    assert wilson_interval(0, 0)[0] != wilson_interval(0, 0)[0]  # nan when n == 0


def test_policy_labels_and_hop_results(src):
    assert [policy_label(r) for r in src.runs] == ["baseline-random", "baseline-oracle"]  # fixture has no config.policy
    assert all(has_hop_results(r) for r in src.runs)


def test_latest_per_policy_dedupes():
    from wikigame.webapp.data import RunInfo

    def mk(i, created):
        return RunInfo(i, i, "local", "g", "baseline", "finished", [], created, {"policy": "random", "graph": "w"}, {"eval/success@1": 0.1})

    kept = latest_per_policy([mk("old", "2026-11-01"), mk("new", "2026-11-05")])
    assert [r.id for r in kept] == ["new"]


def test_baseline_table_has_ci_and_n(src):
    t = baseline_table(src.runs)
    assert set(t.hop) == {1, 2, 3} and (t.ci_low <= t.success).all() and (t.success <= t.ci_high).all()
    assert hop_n(src.runs[0], 1) == src.runs[0].summary["eval/n@1"] == 10  # n@k logged by summarize


def test_comparability_warnings():
    from wikigame.webapp.data import RunInfo

    a = RunInfo("a", "a", "local", "", "baseline", "", [], "", {"graph": "x", "max_steps": 10}, {})
    b = RunInfo("b", "b", "cuda", "", "baseline", "", [], "", {"graph": "y", "max_steps": 20}, {})
    assert len(comparability_warnings([a, b])) == 2 and comparability_warnings([a, a]) == []


# ---- invalid causes ---------------------------------------------------------------------------------------
from wikigame.eval import summarize  # noqa: E402
from wikigame.rollout import oracle_policy  # noqa: E402
from wikigame.webapp.analysis import invalid_cause_table, step_causes, top_bad_titles  # noqa: E402


def _bad_episode(graph):
    """Source A, target H (not linked from A). Answers: target, current page, junk, then a valid move."""
    answers = iter(['<answer>{"title": "H"}</answer>', '<answer>{"title": "A"}</answer>', "junk"])
    pol = lambda o, e: next(answers, None) or oracle_policy()(o, e)  # noqa: E731  (then play on, validly)
    return run_episode(WikiGameEnv(graph, max_invalid=5, seed=0), pol, options={"source": 0, "target": 7}, n_hops=2).to_dict()


def test_step_causes_recorded_and_backfilled_agree(graph):
    ep = _bad_episode(graph)
    recorded = step_causes(None, ep)  # no graph needed when causes were stored
    assert recorded[:3] == ["target_not_linked", "current_page", "unparseable"]
    old = {**ep, "steps": [{k: v for k, v in s.items() if k != "cause"} for s in ep["steps"]]}  # pre-cause format
    assert step_causes(graph, old) == recorded  # backfill reproduces the env's classification
    assert set(step_causes(None, old)) - {None} == {"unrecorded"}  # no graph, no cause: honest about it


def test_invalid_cause_table_and_top_titles(graph):
    steps = steps_df([_bad_episode(graph)], graph)
    t = invalid_cause_table(steps)
    assert t["count"].sum() == (~steps.valid).sum() and abs(t.share_of_invalid.sum() - 1) < 1e-9
    top = top_bad_titles(steps)
    assert set(top.chosen) == {"H", "A"}  # unparseable has no title and is skipped
    assert invalid_cause_table(steps[steps.valid]).empty


def test_summarize_cause_rates_sum_to_invalid_rate(graph):
    eps = [run_episode(WikiGameEnv(graph, max_invalid=3, seed=0), lambda o, e: "x", options={"source": 0, "target": 7}, n_hops=2)]
    m = summarize(eps)
    assert abs(sum(v for k, v in m.items() if k.startswith("invalid_cause/")) - m["invalid_rate"]) < 1e-9


def test_reward_breakdown_reconciles_with_shaping(graph):
    from wikigame.env import RewardConfig
    rc = RewardConfig(shaping=0.5)
    answers = iter(["junk"])
    pol = lambda o, e: next(answers, None) or oracle_policy()(o, e)  # noqa: E731
    ep = run_episode(WikiGameEnv(graph, reward=rc, seed=0), pol, options={"source": 0, "target": 7}).to_dict()
    rb = reward_breakdown([ep], vars(rc))
    assert "progress shaping" in rb.columns and rb.residual.abs().max() < 1e-9


def test_reward_breakdown_reconciles_with_ranked_invalid_tiers(graph):
    from wikigame.env import RewardConfig
    rc = RewardConfig(shaping=0.5)
    answers = iter(["junk", '<answer>{"title": "Nope"}</answer>', '<answer>{"title": "A"}</answer>'])  # 3 different tiers
    pol = lambda o, e: next(answers, None) or oracle_policy()(o, e)  # noqa: E731
    ep = run_episode(WikiGameEnv(graph, reward=rc, max_invalid=5, seed=0), pol, options={"source": 0, "target": 7}).to_dict()
    assert {s["cause"] for s in ep["steps"] if not s["valid"]} == {"unparseable", "unknown_title", "current_page"}
    assert reward_breakdown([ep], vars(rc)).residual.abs().max() < 1e-9
