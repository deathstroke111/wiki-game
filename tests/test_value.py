"""Value task (step 0): strict splits, position sampling, prompt format, ranking metric, and that training learns."""
from pathlib import Path

import pytest

from wikigame import WikiGraph
from wikigame.eval import heldout_pairs
from wikigame.value import (ANSWER_PREFIX, CLASSES, Position, flatten, label, make_splits, ranking_metrics,
                            sample_positions)

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


def test_labels_clip_at_five():
    assert [label(d) for d in (1, 2, 3, 4, 5, 9)] == [0, 1, 2, 3, 4, 4] and CLASSES[label(7)] == "5"


def test_splits_are_target_strict(wiki):
    s = make_splits(wiki, n_val=200, seed=0)
    held = heldout_pairs(wiki)
    assert s.banned == {p for pair in held for p in pair}
    assert set(s.test_targets) == {t for _, t in held}
    assert not (set(s.train_targets) | set(s.val_targets)) & s.banned
    assert not set(s.train_targets) & set(s.val_targets) and len(s.val_targets) == 200


def test_positions_have_true_labels_and_no_banned_pages(wiki):
    s = make_splits(wiki, seed=0)
    pos = sample_positions(wiki, s.train_targets, 40, banned=s.banned, max_negatives=3, seed=1)
    assert len(pos) == 40 and {p.dist for p in pos} <= {2, 3, 4, 5, 6, 7}
    for p in pos:
        d = wiki.dist_to(p.target)
        assert d[p.page] == p.dist and p.link_dists == [d[x] for x in p.links]
        assert all(x in wiki.adj[p.page] for x in p.links)
        assert not ({p.target, p.page} | set(p.links)) & s.banned
        closer = sum(dx < p.dist for dx in p.link_dists)
        assert closer >= 1 and len(p.links) - closer <= 3
    full = sample_positions(wiki, s.val_targets, 5, banned=s.banned, max_negatives=None, seed=2)
    for p in full:  # ranking eval keeps every reachable, non-banned link
        d = wiki.dist_to(p.target)
        assert sorted(p.links) == sorted(x for x in wiki.adj[p.page] if x in d and x not in s.banned)
    assert len(flatten(pos)) == sum(len(p.links) for p in pos)


def test_prompt_has_target_summary_and_answers_without_reasoning(wiki, tok):
    from wikigame.value import class_token_ids, prompt_text

    t, x = 10, wiki.adj[10][0] if wiki.adj[10] else 11
    text = prompt_text(tok, wiki, x, t)
    assert wiki.summaries[t][:40] in text and f'"{wiki.titles[x]}"' in text
    assert text.endswith("<think>\n" + ANSWER_PREFIX) and len(class_token_ids(tok)) == 5


def test_ranking_metrics_perfect_reversed_and_ties():
    pos = [Position(0, 1, 2, [10, 11, 12], [1, 2, 3]), Position(0, 2, 3, [13, 14], [2, 3])]
    good = [[(1.0, 0), (2.0, 1), (3.0, 2)], [(2.0, 1), (3.0, 2)]]
    bad = [[(3.0, 2), (2.0, 1), (1.0, 0)], [(3.0, 1), (2.0, 2)]]
    flat = [[(2.0, 1)] * 3, [(2.0, 1)] * 2]
    g, b, f = ranking_metrics(pos, good), ranking_metrics(pos, bad), ranking_metrics(pos, flat)
    assert g["top1_closer@2"] == 1.0 and g["auc@2"] == 1.0 and g["accuracy"] == 1.0
    assert b["top1_closer@2"] == 0.0 and b["auc@3"] == 0.0
    assert f["top1_closer@2"] == pytest.approx(1 / 3) and f["auc@2"] == 0.5  # ties split evenly
    assert g["random_closer@2"] == pytest.approx(1 / 3) and g["random_closer@3"] == pytest.approx(0.5)


def test_value_training_lowers_the_loss(wiki, tok):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from wikigame.rl.learner import Learner
    from wikigame.value import class_logits, class_token_ids, prompt_text, score_positions, value_loss

    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                   tie_word_embeddings=True)
    L = Learner(R1, model=transformers.AutoModelForCausalLM.from_config(cfg), tok=tok, device="cpu",
                dtype=torch.float32, lr=1e-2, total_updates=70, lr_schedule="constant")
    s = make_splits(wiki, seed=0)
    pos = sample_positions(wiki, s.train_targets, 4, banned=s.banned, max_negatives=2, seed=3)
    qs = flatten(pos)
    ids = class_token_ids(tok)
    prompts = [tok(prompt_text(tok, wiki, x, t), add_special_tokens=False).input_ids for x, t, _ in qs]
    labels = torch.tensor([c for _, _, c in qs])
    first = None
    for _ in range(70):
        loss, _ = value_loss(class_logits(L, prompts, ids), labels)
        loss.backward()
        L.optimizer_step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.8 * first  # measured: 1.61 -> ~1.18, accuracy 0.6 -> 0.9 (tiny random model)
    scores = score_positions(L, tok, wiki, pos, batch=8)
    assert [len(sc) for sc in scores] == [len(p.links) for p in pos] and all(1 <= e <= 5 for sc in scores for e, _ in sc)


# ---- step 1: value-ranked play ------------------------------------------------------------------------------
def _true_value(graph, sign=1.0):
    cache = {}

    def score(pairs):
        out = []
        for x, t in pairs:
            d = cache.setdefault(t, graph.dist_to(t))
            out.append(sign * min(d.get(x, 9), 9))
        return out
    return score


def test_perfect_value_wins_every_game_on_a_shortest_path(wiki):
    from wikigame.eval import make_pairs
    from wikigame.value import play_value_games

    pool = make_pairs(wiki, hops=(2, 3, 4), n_per_hop=50)
    games = [(s, t, k) for k, pl in pool.items() for s, t in pl[:4]]
    eps = play_value_games(wiki, _true_value(wiki), games, batch_size=5)
    assert len(eps) == 12 and all(e.success for e in eps)
    assert sum(e.extra_steps for e in eps) <= 1  # at most a hidden link (200-link cap) costs a detour
    assert all(e.n_invalid == 0 for e in eps) and [e.n_hops for e in eps] == [k for _, _, k in games]
    bad = play_value_games(wiki, _true_value(wiki, sign=-1.0), games, batch_size=5)
    assert sum(e.success for e in bad) <= 2  # always walking away from the target rarely arrives


def test_value_choice_clicks_a_linked_target_and_breaks_ties_at_random(wiki):
    import random as _r

    from wikigame.env import parse_action
    from wikigame.value import value_choice

    t = 5
    links = [wiki.titles[x] for x in (1, 2, 3)]
    title, text = value_choice(wiki, t, links + [wiki.titles[t]], [3.0, 1.0, 2.0, 0.0], _r.Random(0))
    assert title == wiki.titles[t] and parse_action(text) == title
    picks = {value_choice(wiki, t, links, [1.0, 1.0, 2.0], _r.Random(s))[0] for s in range(20)}
    assert picks == {links[0], links[1]}


def test_value_scorer_returns_distances_and_caches(wiki, tok):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from wikigame.value import ValueScorer

    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                   tie_word_embeddings=True)
    sc = ValueScorer(wiki, R1, None, device="cpu", batch=4, model=transformers.AutoModelForCausalLM.from_config(cfg),
                     tok=tok)
    pairs = [(wiki.adj[0][0], 7), (wiki.adj[0][1], 7), (3, 9)]
    e = sc(pairs)
    assert len(e) == 3 and all(1.0 <= v <= 5.0 for v in e) and sc.questions == 3
    assert sc(pairs[:2]) == e[:2] and sc.questions == 3  # cached: nothing re-scored


def test_eval_pairs_for_a_hop_subset_are_the_standard_held_out_pairs(wiki):
    from wikigame.eval import EVAL_POOL, make_pairs

    full = make_pairs(wiki, hops=(1, 2, 3, 4), n_per_hop=EVAL_POOL)
    sub = make_pairs(wiki, hops=(2, 3, 4), n_per_hop=EVAL_POOL)
    assert all(sub[h] == full[h] for h in (2, 3, 4))
    assert {p for ps in full.values() for p in ps} == heldout_pairs(wiki)
