from wikigame import WikiGameEnv, WikiGraph
from wikigame.env import SYSTEM_PROMPT, parse_action
from wikigame.llm import HFPolicy, build_messages
from wikigame.rollout import run_episode


def test_build_messages_structure():
    m = build_messages("OBS")
    assert [x["role"] for x in m] == ["system", "user"] and m[0]["content"] == SYSTEM_PROMPT and m[1]["content"] == "OBS"


def test_policy_plugs_into_run_episode_without_loading_a_model():
    """HFPolicy only needs .generate(messages); stub it so the test never downloads weights."""
    g = WikiGraph.from_edges({t: "s" for t in "ABC"}, {"A": ["B"], "B": ["C"], "C": []})

    class Stub(HFPolicy):
        def __init__(self):
            self.seen = []

        def generate(self, messages):
            self.seen.append(messages)
            links = [l[2:] for l in messages[1]["content"].split("Links:\n")[1].splitlines()]
            return f'<think>go</think><answer>{{"title": "{links[0]}"}}</answer>'

    pol = Stub()
    ep = run_episode(WikiGameEnv(g, seed=0), pol, options={"source": 0, "target": 2})
    assert ep.success and ep.path == ["A", "B", "C"] and len(pol.seen) == 2
    assert parse_action(ep.steps[0]["text"]) == "B"


def test_lenient_answer_rescues_bare_and_fenced_json_only():
    from wikigame.llm import lenient_answer

    good = '<think>x</think><answer>{"title": "A"}</answer>'
    assert lenient_answer(good) == good  # untouched
    fenced = '```json\n{"title": "Sanskrit \\"x\\""}\n```'
    out = lenient_answer(fenced)
    assert out.startswith(fenced) and parse_action(out) == 'Sanskrit "x"'  # raw kept, title recovered
    assert parse_action(lenient_answer('{"title": "B", "reason": "r"}')) == "B"
    assert lenient_answer("I think China.") == "I think China."  # no title -> unchanged (still invalid)
    assert parse_action(lenient_answer("I think China.")) is None
