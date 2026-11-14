"""The v3 training loop end to end on CPU: a tiny random Qwen2 with the real R1 tokenizer (from the local HF cache),
scripted answers instead of sampling, the real env/reward/state pool. Skipped when the tokenizer is not cached."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from wikigame import WikiGraph  # noqa: E402
from wikigame.grpo_reward import StepReward  # noqa: E402
from wikigame.onpolicy import StatePool  # noqa: E402
from wikigame.rl.algorithms import get_algorithm  # noqa: E402
from wikigame.rl.generation import HFGenerator, ScriptedGenerator, encode_prompt  # noqa: E402
from wikigame.rl.learner import Learner  # noqa: E402
from wikigame.rl.sources import OnPolicyRounds, StepSource  # noqa: E402
from wikigame.rl.trainer import RLTrainer, TrainConfig  # noqa: E402
from wikigame.states import StateMix, sample_states  # noqa: E402

R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CACHE = Path(__file__).resolve().parents[1] / "data" / "hf" / "hub"


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained(R1, cache_dir=str(CACHE), local_files_only=True)
    except OSError:
        pytest.skip("R1 tokenizer not in data/hf")


@pytest.fixture(scope="module")
def graph():
    n = 40
    adj = {f"P{i}": [f"P{(i + 1) % n}", f"P{(i + 2) % n}", f"P{(i * 7 + 3) % n}"] for i in range(n)}
    return WikiGraph.from_edges({t: f"About {t}" for t in adj}, adj)


def tiny_model(tok):
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                   tie_word_embeddings=True)
    return transformers.AutoModelForCausalLM.from_config(cfg)


def learner(tok, **kw):
    return Learner(R1, model=tiny_model(tok), tok=tok, device="cpu", dtype=torch.float32, lr=1e-2, **kw)


def ans(title):
    return f'x</think><answer>{{"title": "{title}"}}</answer>'


def oracle_or_garbage(graph, tok):
    """k even: the BFS-best link; k odd: an unparseable answer. Gives every group reward spread."""
    def answer(prompt_ids, k):
        text = tok.decode(prompt_ids, skip_special_tokens=True)
        target = text.split("Target: ", 1)[1].splitlines()[0]
        lines = text.split("Links:\n", 1)[1].splitlines()
        links = [l[2:].split("<｜")[0] for l in lines if l.startswith("- ")]  # R1's template tags are not "special"
        if k % 2:
            return "I am not sure"
        d = graph.dist_to(graph.id_of(target))
        return ans(min(links, key=lambda t: d.get(graph.id_of(t), 10**9)))
    return answer


def make(graph, tok, tmp_path, algo="grpo", gen=None, L=None, **cfg):
    rows = sample_states(graph, 24, R1, StateMix(hops=(2, 3)), seed=0)
    pool = StatePool(rows, share=0.5, seed=0)
    L = L or learner(tok, total_updates=10)
    gen = gen or ScriptedGenerator(tok, oracle_or_garbage(graph, tok))
    src = StepSource(pool, tok, gen, StepReward(graph, max_completion_length=64), num_generations=4, max_tokens=64,
                     temperature=1.0)
    c = TrainConfig(out=tmp_path / "out", max_steps=2, prompts_per_step=2, num_generations=4, micro_batch=3,
                    max_completion=64, save_steps=2, **cfg)
    return RLTrainer(L, gen, src, get_algorithm(algo), c, sync_dir=tmp_path / "sync"), L, gen, pool


def lora_state(L):
    return {k: v.detach().clone() for k, v in L.model.named_parameters() if v.requires_grad}


def test_padded_logprobs_match_one_sequence_at_a_time(tok):
    L = learner(tok, temperature=0.7)
    prompts = [tok("short prompt").input_ids, tok("a somewhat longer prompt than the first").input_ids]
    comps = [tok(" answer one").input_ids, tok(" two").input_ids]
    lp, mask, ent = L.token_logprobs(prompts, comps, with_entropy=True)
    assert mask.tolist()[1][len(comps[1]):] == [0.0] * (lp.shape[1] - len(comps[1]))
    for i, (p, c) in enumerate(zip(prompts, comps)):
        ids = torch.tensor([p + c])
        logits = L.model(input_ids=ids).logits[0, len(p) - 1:-1] / 0.7
        ref = torch.log_softmax(logits.float(), -1).gather(-1, torch.tensor(c)[:, None]).squeeze(-1)
        assert torch.allclose(lp[i, :len(c)], ref, atol=1e-4)
    assert torch.all(ent[mask.bool()] > 0)


def test_training_steps_update_the_lora_and_save(graph, tok, tmp_path):
    tr, L, gen, _ = make(graph, tok, tmp_path)
    gen.wants_adapter_files = True  # exercise the vLLM hand-over path
    before = lora_state(L)
    m = tr.run()
    after = lora_state(L)
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert m["frac_reward_zero_std"] == 0 and m["reward_std"] > 0 and m["train/sequences"] == 8
    assert m["clip_ratio/region_mean"] == 0 and abs(m["ratio_mean"] - 1) < 1e-6  # on-policy: ratio exactly 1
    assert 0 < m["env/invalid_rate"] < 1
    out = tmp_path / "out"
    assert (out / "adapter_model.safetensors").exists() and (out / "checkpoint-2" / "optimizer.pt").exists()
    assert len((out / "completions.jsonl").read_text().splitlines()) == 2 * 2 * 4
    assert len(gen.adapters) == 2 and all((Path(p) / "adapter_model.safetensors").exists() for p in gen.adapters)
    L2 = learner(tok, total_updates=10)  # resume: optimizer state loads into a fresh learner
    L2.load_optimizer(out / "checkpoint-2")
    assert L2.updates == 2


def test_batch_reuse_moves_the_ratio_and_engages_the_clip(graph, tok, tmp_path):
    tr, *_ = make(graph, tok, tmp_path, algo="dapo", updates_per_batch=3)
    m = tr.train_step()
    assert m["ratio_mean"] != pytest.approx(1.0, abs=1e-6)
    assert "clip_ratio/high_mean" in m and m["learning_rate"] > 0


def test_zero_std_groups_are_dropped_only_by_dapo(graph, tok, tmp_path):
    same = ScriptedGenerator(tok, lambda p, k: "I am not sure")  # every answer scores the same
    tr, *_ = make(graph, tok, tmp_path, algo="dapo", gen=same)
    m = tr.train_step()
    assert m["frac_reward_zero_std"] == 1 and m.get("skipped_update") == 1.0
    tr2, *_ = make(graph, tok, tmp_path / "b", algo="grpo", gen=ScriptedGenerator(tok, lambda p, k: "I am not sure"))
    assert "skipped_update" not in tr2.train_step()


def test_sampler_logprobs_are_compared_and_tis_applies(graph, tok, tmp_path):
    base = ScriptedGenerator(tok, oracle_or_garbage(graph, tok))
    real = base.generate

    def with_lp(prompt_ids, n, max_tokens, temperature, seed=None):
        out = real(prompt_ids, n, max_tokens, temperature, seed)
        for g in out:
            for s in g:
                s.logprobs = [-1.0] * len(s.token_ids)
        return out
    base.generate = with_lp
    tr, *_ = make(graph, tok, tmp_path, algo="cispo", gen=base, tis_cap=2.0)
    m = tr.train_step()
    assert m["sampler/logprob_abs_diff"] > 0 and 0 <= m["sampler/tis_clipped_ratio"] <= 1


def test_every_preset_runs_a_step(graph, tok, tmp_path):
    for name in ("reinforce", "rloo", "grpo", "dapo", "trl-dapo", "cispo", "gspo"):
        tr, *_ = make(graph, tok, tmp_path / name, algo=name)
        m = tr.train_step()
        assert torch.isfinite(torch.tensor(m["loss"])) and m["grad_norm"] > 0, name


def test_hf_generator_returns_n_samples_with_eos_or_cap(tok):
    L = learner(tok)
    gen = HFGenerator(L.model, tok)
    ids, opened = encode_prompt(tok, [{"role": "user", "content": "hi"}])
    assert opened  # R1's template opens <think>
    out = gen.generate([ids, ids[:5]], 3, 6, 1.0, seed=0)
    assert [len(g) for g in out] == [3, 3]
    for s in (s for g in out for s in g):
        assert len(s.token_ids) <= 6 and (s.finished == (s.token_ids[-1] == tok.eos_token_id) or s.finished)
    assert L.model.training  # generation restores train mode


def test_onpolicy_round_refills_the_pool(graph, tok, tmp_path):
    rows = sample_states(graph, 10, R1, StateMix(hops=(2, 3)), seed=0)
    pool = StatePool(rows, share=1.0, seed=0)
    gen = ScriptedGenerator(tok, lambda p, k: oracle_or_garbage(graph, tok)(p, 0))
    rounds = OnPolicyRounds(pool, graph, gen, tok, every=5, n_games=6, hops=(2, 3), model_id=R1, hint=None,
                            max_steps_game=10, max_tokens=64, temperature=1.0, batch_size=4, feedback=False, seed=0,
                            log_path=tmp_path / "rounds.jsonl")
    m = rounds.play(0)
    assert m["onpolicy/success"] == 1.0 and m["onpolicy/states"] > 0
    assert pool[0]["kind"] == "onpolicy" and set(pool[0]) == set(rows[0])
    assert rounds.due(5) and not rounds.due(3)
    assert len((tmp_path / "rounds.jsonl").read_text().splitlines()) == 6


def test_playout_return_orders_outcomes():
    from wikigame.rl.sources import PlayoutReturn

    r = PlayoutReturn()
    slow_win, fast_win = r.score(True, 10, 3, 0, "success"), r.score(True, 2, 3, 0, "success")
    near_loss, far_loss = r.score(False, 10, 3, 1, "step_limit"), r.score(False, 10, 3, 3, "step_limit")
    invalid_loss = r.score(False, 4, 3, 3, "invalid_action")
    assert fast_win > slow_win >= 0.5 > near_loss > far_loss == 0.0 > invalid_loss


def test_move_bonus_is_signed_and_off_by_default():
    from wikigame.rl.sources import PlayoutReturn

    assert PlayoutReturn().bonus("closer") == 0.0
    r = PlayoutReturn(move_bonus=0.2)
    assert (r.bonus("closer"), r.bonus("farther"), r.bonus("sideways"), r.bonus("invalid")) == (0.2, -0.2, 0.0, 0.0)


def test_move_bonus_adds_to_the_playout_score(graph, tok):
    from wikigame.rl.sources import PlayoutReturn, PlayoutSource

    rows = sample_states(graph, 8, R1, StateMix(hops=(2, 3)), seed=1)
    scores = {}
    for b in (0.0, 0.2):
        gen = ScriptedGenerator(tok, oracle_or_garbage(graph, tok))  # oracle (closer) / garbage (invalid) answers
        src = PlayoutSource(StatePool(rows, share=0.0, seed=0), graph, tok, gen, num_generations=4, max_tokens=64,
                            temperature=1.0, model_id=R1, hint=None, returns=PlayoutReturn(move_bonus=b))
        groups, m = src.next_batch(2)
        scores[b] = [r for g in groups for r in g.rewards]
    # same scripted games both times: oracle answers (even) gain exactly 0.2, invalid ones (odd) nothing
    assert [round(y - x, 6) for x, y in zip(scores[0.0], scores[0.2])] == [0.2, 0.0] * 4
    assert m["playout/bonus_mean"] == pytest.approx(0.1) and "playout/return_mean" in m


def test_playout_source_scores_answers_by_their_finished_games(graph, tok, tmp_path):
    from wikigame.rl.sources import PlayoutSource

    rows = sample_states(graph, 8, R1, StateMix(hops=(2, 3)), seed=1)
    # the branching answer alternates oracle / garbage (k even / odd); play-out moves are always the oracle (k = 0)
    gen = ScriptedGenerator(tok, oracle_or_garbage(graph, tok))
    src = PlayoutSource(StatePool(rows, share=0.0, seed=0), graph, tok, gen, num_generations=4, max_tokens=64,
                        temperature=1.0, model_id=R1, hint=None)
    groups, m = src.next_batch(2)
    assert len(groups) == 2 and all(len(g.rewards) == 4 for g in groups)
    assert m["playout/success"] == 1.0 and m["env/invalid_rate"] == 0.5 and m["playout/calls"] >= 1
    # oracle answers always move closer, garbage ones are invalid (closer was always 0 before 2026-11-14)
    assert m["env/closer_rate"] == 0.5 and m["env/sideways_rate"] == 0.0
    assert sum(m[f"env/{d}/share"] for d in ("d1", "d2", "d3", "d4p")) == pytest.approx(1.0)
    assert all(m[k] == 0.5 for k in m if k.endswith("/closer_rate") and k.startswith("env/d"))
    for g in groups:  # a garbage first answer wastes a move, so its (won) game scores lower
        assert g.rewards[0] > g.rewards[1] and g.rewards[2] > g.rewards[3]
    tr, L, _, _ = make(graph, tok, tmp_path)
    tr.source = src
    m2 = tr.train_step()
    assert m2["reward_std"] > 0 and m2["grad_norm"] > 0


def test_lr_schedule_warmup_cosine_and_floor():
    from wikigame.rl.learner import _schedule

    f = _schedule("cosine", 300, 10, 0.1)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)  # linear warm-up over 10 updates
    assert f(10) == pytest.approx(1.0) and f(155) == pytest.approx(0.55)  # cosine midpoint: halfway to the floor
    assert f(299) == pytest.approx(0.1, abs=1e-3) and f(400) == pytest.approx(0.1)  # never below the floor
    assert all(f(i) >= f(i + 1) for i in range(10, 299))
    lin = _schedule("linear", 300, 0)  # the stage 1-5 default is unchanged: 1 -> 0
    assert lin(0) == 1.0 and lin(150) == pytest.approx(0.5) and lin(300) == 0.0
    with pytest.raises(ValueError):
        _schedule("cosine", 300, 0, 1.5)


def test_selective_logprobs_match_log_softmax_in_value_and_gradient():
    from wikigame.rl.learner import selective_logprobs

    torch.manual_seed(0)
    logits = torch.randn(3, 5, 17, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 17, (3, 5))
    w = torch.randn(3, 5, dtype=torch.float64)
    ref = torch.log_softmax(logits / 0.7, -1).gather(-1, targets[..., None]).squeeze(-1)
    (ref * w).sum().backward()
    g_ref = logits.grad.clone()
    logits.grad = None
    lp, ent = selective_logprobs(logits, targets, 1 / 0.7, with_entropy=True)
    (lp * w).sum().backward()
    assert torch.allclose(lp.double(), ref, atol=1e-5) and torch.allclose(logits.grad, g_ref, atol=1e-5)
    p = torch.softmax(logits.detach() / 0.7, -1)
    assert torch.allclose(ent.double(), -(p * p.log()).sum(-1), atol=1e-4)
