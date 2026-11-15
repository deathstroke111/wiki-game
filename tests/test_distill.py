"""Value step 2 (distillation): reasoning choice, strict states, answer-only loss, and that it learns."""
from pathlib import Path

import pytest

from wikigame import WikiGraph
from wikigame.distill import THINK_END, answer_text, choose_reasoning, move_example, sample_answer, strict_states
from wikigame.value import make_splits

ROOT = Path(__file__).resolve().parents[1]
R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@pytest.fixture(scope="module")
def wiki():
    p = ROOT / "data" / "wikispeedia.json"
    if not p.exists():
        pytest.skip("data/wikispeedia.json not built")
    return WikiGraph.load(p)


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(R1, cache_dir=str(ROOT / "data" / "hf" / "hub"),
                                                          local_files_only=True)
    except OSError:
        pytest.skip("R1 tokenizer not in data/hf")


def raw(reason: str, title: str | None) -> str:  # a sample as vLLM returns it (after the template's "<think>\n")
    return f"{reason}\n</think>\n\n" + (f'<answer>{{"title": "{title}"}}</answer>' if title else "no answer")


def test_choose_reasoning_prefers_a_matching_sample_then_swaps_then_drops():
    s = [raw("go via Asia", "Asia"), raw("Europe is closer", "Europe"), "cut off mid-thought about Asia"]
    assert sample_answer(s[1]) == "Europe"
    r, matched = choose_reasoning(s, "Europe")
    assert matched and r == "Europe is closer\n</think>\n\n"
    r, matched = choose_reasoning(s, "Africa")  # nobody chose the label: first finished reasoning, answer swapped
    assert not matched and r.startswith("go via Asia") and r.endswith(THINK_END + "\n\n")
    assert choose_reasoning(["still thinking", "and thinking"], "Asia") is None


def test_strict_states_use_only_allowed_targets_and_no_banned_pages(wiki):
    sp = make_splits(wiki, seed=0)
    rows = strict_states(wiki, 30, R1, sp, set(sp.train_targets), seed=0)
    assert 20 <= len(rows) <= 30
    assert all(r["target"] in set(sp.train_targets) for r in rows)
    assert not any(p in sp.banned for r in rows for p in r["path"])
    assert all(wiki.titles[r["path"][-1]] in r["prompt"][-1]["content"] for r in rows)


def test_move_example_masks_everything_but_the_answer(tok):
    p, c, start = move_example(tok, [1, 2, 3], "I think Asia.\n</think>\n\n", "Asia")
    assert p == [1, 2, 3] and c[-1] == tok.eos_token_id
    assert tok.decode(c[:start]) == "I think Asia.\n</think>\n\n"
    assert tok.decode(c[start:-1]) == answer_text("Asia")


def test_answer_training_learns_the_label_after_the_reasoning(tok):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import importlib.util

    from wikigame.rl.learner import Learner

    spec = importlib.util.spec_from_file_location("distill_train", ROOT / "scripts" / "distill_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                   tie_word_embeddings=True, initializer_range=0.5)
    # initializer_range 0.5: with the default 0.02 the tiny tied embeddings cap every logit near 0, so the NLL over
    # a 152k vocabulary cannot fall below ~ln(152k) - 0.6 whatever the LoRA does (measured plateau 11.4)
    L = Learner(R1, model=transformers.AutoModelForCausalLM.from_config(cfg), tok=tok, device="cpu",
                dtype=torch.float32, lr=1e-2, total_updates=60, lr_schedule="constant", temperature=1.0)
    prompt = tok("Target: Asia. Links: Asia, Europe.", add_special_tokens=False).input_ids
    exs = [move_example(tok, prompt, "Thinking about it.\n</think>\n\n", t) for t in ("Asia", "Asia", "Asia")]
    first = None
    for _ in range(60):
        loss = mod.answer_nll(L, exs).mean()
        loss.backward()
        L.optimizer_step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.5 * first
