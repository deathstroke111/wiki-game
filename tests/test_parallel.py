"""Parallel game player and on-policy state pool: driven by scripted stubs, so no model or torch is needed."""
import pytest

from wikigame import WikiGameEnv, WikiGraph
from wikigame.grpo_reward import StepReward
from wikigame.llm import training_messages
from wikigame.onpolicy import StatePool, round_pairs, select_states
from wikigame.parallel import play_games
from wikigame.rollout import oracle_policy, run_episode

R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@pytest.fixture(scope="module")
def graph():
    n = 40
    adj = {f"P{i}": [f"P{(i + 1) % n}", f"P{(i + 2) % n}", f"P{(i * 7 + 3) % n}"] for i in range(n)}
    return WikiGraph.from_edges({t: f"About {t}" for t in adj}, adj)


def ans(title):
    return f'<think>x</think><answer>{{"title": "{title}"}}</answer>'


def parse_prompt(msgs):
    obs = msgs[-1]["content"]
    target = obs.split("Target: ", 1)[1].splitlines()[0]
    links = [l[2:] for l in obs.split("Links:\n", 1)[1].splitlines()]
    return target, links


def oracle_stub(graph, log=None):
    def generate(batch):
        if log is not None:
            log.append(len(batch))
        out = []
        for msgs in batch:
            target, links = parse_prompt(msgs)
            d = graph.dist_to(graph.id_of(target))
            out.append(ans(min(links, key=lambda t: d.get(graph.id_of(t), 10**9))))
        return out
    return generate


GAMES = [(0, 5, 3), (1, 3, 1), (4, 10, 3), (7, 9, 1), (2, 14, 4), (11, 13, 1), (20, 26, 3)]


def test_oracle_games_match_one_at_a_time_play(graph):
    sizes = []
    eps, rows = play_games(graph, oracle_stub(graph, sizes), GAMES, batch_size=3)
    assert [(e.source, e.target, e.n_hops) for e in eps] == [(f"P{s}", f"P{t}", k) for s, t, k in GAMES]
    assert max(sizes) <= 3 and sizes[0] == 3  # never more than batch_size games per call
    env = WikiGameEnv(graph, seed=0)
    for e, (s, t, k) in zip(eps, GAMES):
        ref = run_episode(env, oracle_policy(), options={"source": s, "target": t}, n_hops=k)
        assert e.success and len(e.path) == len(ref.path) and e.extra_steps == 0
    assert len(rows) == sum(e.n_valid for e in eps)  # one decision per move, every move valid


def test_finished_slot_goes_to_the_next_game_and_callback_fires_once(graph):
    seen = []
    eps, _ = play_games(graph, oracle_stub(graph), GAMES, batch_size=2, on_episode=lambda i, e: seen.append(i))
    assert sorted(seen) == list(range(len(GAMES))) and all(eps)


def test_garbage_answers_end_each_game_after_three_invalids(graph):
    eps, rows = play_games(graph, lambda b: ["I am not sure"] * len(b), GAMES[:3], batch_size=8)
    assert all(e.end_reason == "invalid_action" and e.n_invalid == 3 and e.steps[0]["cause"] == "unparseable"
               for e in eps)
    assert [r["n_invalid"] for r in rows if r["game"] == 0] == [0, 1, 2]
    assert all(r["path"] == [r["source"]] for r in rows)


def test_lenient_rescues_bare_json(graph):
    def bare(batch):
        return ['{"title": "%s"}' % parse_prompt(m)[1][0] for m in batch]

    strict, _ = play_games(graph, bare, GAMES[:2])
    lenient, _ = play_games(graph, bare, GAMES[:2], lenient=True)
    assert all(e.n_valid == 0 for e in strict) and all(e.n_valid > 0 for e in lenient)


def test_rows_are_training_prompts_and_replay_in_the_reward(graph):
    _, rows = play_games(graph, oracle_stub(graph), GAMES, messages_fn=lambda o: training_messages(o, R1, None))
    assert all([m["role"] for m in r["prompt"]] == ["user"] for r in rows)  # R1 format, as GRPO trains it
    reward = StepReward(graph)
    mid = next(r for r in rows if len(r["path"]) > 1)
    target, links = parse_prompt(mid["prompt"])
    d = graph.dist_to(mid["target"])
    best = min(links, key=lambda t: d.get(graph.id_of(t), 10**9))
    r, info = reward.score(ans(best), mid["source"], mid["target"], mid["path"], mid["n_invalid"])
    assert info["valid"] and info["progress"] == 1
    assert f"Current article: {graph.titles[mid['path'][-1]]}" in mid["prompt"][0]["content"]


def test_select_states_dedupes_and_drops_hopeless_positions():
    # B -> C is one-way and C is a dead end: from C the target A can never be reached
    g = WikiGraph.from_edges({t: t for t in "ABCD"}, {"A": ["B"], "B": ["C", "D"], "C": [], "D": ["A"]})
    row = lambda path, n_inv=0: {"source": 1, "target": 0, "path": path, "n_invalid": n_inv}
    kept = select_states(g, [row([1]), row([1], 1), row([1, 3]), row([1, 2])])
    assert [(r["path"], r["n_invalid"]) for r in kept] == [([1], 0), ([1, 3], 0)]


def test_state_pool_keeps_a_row_per_index_and_mixes_at_the_share():
    scripted = [{"kind": "scripted", "i": i} for i in range(400)]
    pool = StatePool(scripted, share=0.5, seed=0)
    assert pool[3]["kind"] == "scripted"  # no round yet: scripted only
    pool.refill([{"kind": "onpolicy", "i": i} for i in range(1000)])
    first = [pool[i] for i in range(4, 304)]
    assert [pool[i] for i in range(4, 304)] == first  # same index -> same state (the G answers of a group)
    share = sum(r["kind"] == "onpolicy" for r in first) / len(first)
    assert 0.4 < share < 0.6
    stats = pool.refill([{"kind": "onpolicy", "i": -1}])
    assert stats["onpolicy/drawn_share"] == pytest.approx(share) and stats["onpolicy/unused_from_last_round"] > 800


def test_state_pool_falls_back_to_scripted_when_a_round_runs_out():
    pool = StatePool([{"kind": "scripted"}] * 10, share=1.0, seed=0)
    pool.refill([{"kind": "onpolicy"}] * 2)
    assert [pool[i]["kind"] for i in range(5)] == ["onpolicy"] * 2 + ["scripted"] * 3


def test_state_pool_rows_share_the_scripted_columns(graph):
    # regression (first pod run): play_games rows carry an extra "game" key; TRL indexes every row of a batch with
    # the first row's keys, so a mixed batch raised KeyError: 'game' in _calculate_rewards
    from wikigame.states import sample_states

    scripted = sample_states(graph, 6, R1, seed=0)
    _, rows = play_games(graph, oracle_stub(graph), GAMES, messages_fn=lambda o: training_messages(o, R1, None))
    assert "game" in rows[0]
    pool = StatePool(scripted, share=0.5, seed=0)
    pool.refill(select_states(graph, rows))
    batch = [pool[i] for i in range(len(scripted))]
    kinds = {r["kind"] for r in batch}
    assert "onpolicy" in kinds and kinds - {"onpolicy"}  # a mixed batch
    assert all(set(r) == set(scripted[0]) for r in batch)
    with pytest.raises(KeyError):
        pool.refill([{"kind": "onpolicy"}])  # a missing column fails at refill, not mid-step


def test_round_pairs_avoid_heldout_pairs(graph):
    import random

    held = {(0, 1), (0, 2)}
    games = round_pairs(graph, 200, (1, 2), random.Random(0), held)
    assert len(games) == 200 and not {(s, t) for s, t, _ in games} & held
    assert all(graph.shortest_dist(s, t) == k for s, t, k in games)
