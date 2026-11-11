"""State sampler + reward function: everything GRPO needs except the model (no downloads, no torch needed)."""
import random

import pytest

from wikigame import WikiGameEnv, WikiGraph
from wikigame.env import RewardConfig
from wikigame.eval import heldout_pairs
from wikigame.grpo_reward import StepReward, default_reward_config
from wikigame.states import StateMix, sample_states

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def graph():
    # a chain of 12 pages with a few side links, so 1..3-hop pairs exist in plenty
    n = 40
    adj = {f"P{i}": [f"P{(i + 1) % n}", f"P{(i + 2) % n}", f"P{(i * 7 + 3) % n}"] for i in range(n)}
    return WikiGraph.from_edges({t: f"About {t}" for t in adj}, adj)


def ans(title):
    return f'<think>x</think><answer>{{"title": "{title}"}}</answer>'


def test_set_state_matches_a_real_game(graph):
    env = WikiGameEnv(graph, seed=0)
    env.reset(options={"source": 0, "target": 20})
    first = env.links[0]
    env.step(first)
    live_obs = env._obs()
    other = WikiGameEnv(graph, seed=0)
    st_obs = other.set_state(0, 20, list(env.path))
    assert other.cur == env.cur and other.visited == env.visited and other.steps == env.steps == 1
    assert sorted(l for l in live_obs.splitlines() if l.startswith("- ")) == sorted(l for l in st_obs.splitlines() if l.startswith("- "))
    assert "Steps left: 9" in st_obs and "Path so far: P0 -> " in st_obs


def test_sample_states_distinct_heldout_and_replayable(graph):
    rows = sample_states(graph, 30, MODEL, StateMix(hops=(1, 2, 3)), seed=1)
    assert len(rows) == 30 and {r["kind"] for r in rows} >= {"start", "optimal"}
    held = heldout_pairs(graph)
    assert all((r["source"], r["target"]) not in held for r in rows)
    keys = {(r["source"], r["target"], tuple(r["path"])) for r in rows}
    assert len(keys) == 30
    r = rows[0]
    env = WikiGameEnv(graph, seed=0)
    assert env.set_state(r["source"], r["target"], r["path"]) and env.cur == r["path"][-1]
    assert r["prompt"][0]["role"] == "system" and r["prompt"][-1]["role"] == "user"  # Qwen keeps the system turn
    assert all(r["path"][-1] != r["target"] for r in rows)  # never a finished game


def test_r1_prompts_have_no_system_turn(graph):
    rows = sample_states(graph, 3, "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", seed=1)
    assert all([m["role"] for m in r["prompt"]] == ["user"] for r in rows) and "briefly" in rows[0]["prompt"][0]["content"]


def test_reward_ordering_on_real_states(graph):
    rows = sample_states(graph, 40, MODEL, StateMix(start=1, optimal=0, noisy=0, hops=(2, 3)), seed=2)
    reward = StepReward(graph)
    checked = 0
    for r in rows:
        env = WikiGameEnv(graph)
        env.set_state(r["source"], r["target"], r["path"])
        d = graph.dist_to(r["target"])
        cur = r["path"][-1]
        cands = [v for v in graph.adj[cur] if v not in r["path"]]
        closer = [v for v in cands if d.get(v, 99) < d[cur]]
        farther = [v for v in cands if d.get(v, 99) > d[cur]]
        if not (closer and farther):
            continue
        args = (r["source"], r["target"], r["path"], 0)
        rc = reward.score(ans(graph.titles[closer[0]]), *args)[0]
        rf = reward.score(ans(graph.titles[farther[0]]), *args)[0]
        ri = reward.score("no tags at all", *args)[0]
        rt = reward.score(ans(graph.titles[cur]), *args)[0]  # re-picking the current page: real page, not a legal move
        ru = reward.score(ans("No such page"), *args)[0]
        assert rc > rf > rt > ru > ri, (rc, rf, rt, ru, ri)
        checked += 1
    assert checked >= 5


def test_reward_call_logs_metrics_and_cutoffs(graph):
    rows = sample_states(graph, 4, MODEL, seed=3)
    logged = {}
    sr = StepReward(graph, max_completion_length=10)
    comps = [[{"role": "assistant", "content": c}] for c in ["no answer, rambling", ans("Nope"), "x", ans("Nope")]]
    out = sr(prompts=[r["prompt"] for r in rows], completions=comps, source=[r["source"] for r in rows],
             target=[r["target"] for r in rows], path=[r["path"] for r in rows], n_invalid=[0] * 4,
             completion_ids=[[0] * 10, [0] * 3, [0] * 10, [0] * 2], log_metric=lambda k, v: logged.__setitem__(k, v))
    # unparseable and cut off: -2.0 -0.25 -0.02 ; non-existent title, finished in time: -1.75 -0.02
    assert [round(o, 2) for o in out] == [-2.27, -1.77, -2.27, -1.77]
    assert logged["env/invalid_rate"] == 1.0 and logged["env/cutoff_rate"] == 0.5  # two ran to the cap w/o </answer>
    assert logged["env/invalid/unparseable"] == 0.5 and logged["env/invalid/unknown_title"] == 0.5


def test_training_reward_is_shaped_eval_reward_is_not():
    assert default_reward_config().shaping > 0 and RewardConfig().shaping == 0


def test_reward_call_logs_rates_split_by_distance(graph):
    rows = sample_states(graph, 12, MODEL, StateMix(hops=(1, 2, 3)), seed=4)
    logged = {}
    sr = StepReward(graph, RewardConfig(shaping=1.0, budget_aware=True, sideways_bonus=0.3))
    comps = [ans(graph.titles[r["links"] and graph.id_of(r["links"][0])]) for r in rows]
    sr(prompts=None, completions=comps, source=[r["source"] for r in rows], target=[r["target"] for r in rows],
       path=[r["path"] for r in rows], n_invalid=[0] * len(rows), log_metric=lambda k, v: logged.__setitem__(k, v))
    shares = {k: v for k, v in logged.items() if k.endswith("/share")}
    assert shares and abs(sum(shares.values()) - 1) < 1e-9 and set(shares) <= {f"env/{b}/share" for b in ("d1", "d2", "d3", "d4p")}
    assert all(f"env/{k.split('/')[1]}/closer_rate" in logged and f"env/{k.split('/')[1]}/invalid_rate" in logged for k in shares)
    assert {"env/sideways_rate", "env/out_of_reach_rate"} <= set(logged) and logged["env/invalid_rate"] == 0


def test_hint_names_resolve_and_reach_the_training_prompt(graph):
    from wikigame.llm import HINTS, R1_HINT, resolve_hint
    assert HINTS["brief"] == R1_HINT and resolve_hint("plan") == HINTS["plan"]
    assert resolve_hint(None) is None and resolve_hint("") == "" and resolve_hint("my own text") == "my own text"
    assert all("Do NOT go through the links one by one" in h for h in HINTS.values())
    r1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    text = sample_states(graph, 2, r1, seed=1, hint=resolve_hint("plan"))[0]["prompt"][0]["content"]
    assert "two moves ahead" in text and "briefly" not in text
    assert sample_states(graph, 2, r1, seed=1)[0]["prompt"] == sample_states(graph, 2, r1, seed=1, hint=HINTS["brief"])[0]["prompt"]


def test_probe_training_states_avoid_heldout_pairs(graph):
    from wikigame.probe import training_states
    states = training_states(graph, [2, 3], 3, seed=0)
    held = heldout_pairs(graph)
    assert states and all(min(s["dist"], 3) in (2, 3) for s in states)
    assert all((graph.id_of(s["path"][0]), graph.id_of(s["target"])) not in held for s in states)
    assert states == training_states(graph, [2, 3], 3, seed=0)
