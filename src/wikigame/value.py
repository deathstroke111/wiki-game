"""Value task (step 0 of the 'internalised value' plan, 2026-11-15): "how many clicks from page X to the target?"

The model answers with one digit (1-5, 5 = five or more) straight after an empty ``<think></think>`` block, so a page's
links are scored in one batched forward pass (no generation): the softmax over the five digit tokens gives an
expected distance, and the link with the smallest one is the value-ranked move.

Held-out strictness (the user's choice): no training question involves any page that appears in a held-out eval pair
(as source or target), in either role. Training targets also exclude a validation set used to pick the checkpoint.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from .eval import heldout_pairs
from .graph import WikiGraph

CLASSES = ["1", "2", "3", "4", "5"]  # 5 = five or more clicks
QUESTION = ("Wikipedia game: you will navigate by clicking links between Wikipedia articles.\n"
            "Target: {target}\nTarget summary: {summary}\n\n"
            'How many clicks does it take to get from the article "{page}" to the target? '
            "Answer with one number: 1, 2, 3, 4 or 5 (5 means five or more).")
ANSWER_PREFIX = "\n</think>\n\n"  # closes the <think> the R1 chat template opens: answer without reasoning


def label(dist: int) -> int:
    """Class index 0..4 for a distance >= 1."""
    return min(dist, 5) - 1


def prompt_text(tok, graph: WikiGraph, page: int, target: int) -> str:
    q = QUESTION.format(target=graph.titles[target], summary=graph.summaries[target], page=graph.titles[page])
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) \
        + ANSWER_PREFIX


def class_token_ids(tok) -> list[int]:
    ids = [tok.encode(c, add_special_tokens=False) for c in CLASSES]
    assert all(len(i) == 1 for i in ids), f"each class must be one token: {ids}"
    return [i[0] for i in ids]


@dataclass
class Splits:
    banned: set[int]  # every page in a held-out eval pair: never in a training or validation question
    test_targets: list[int]  # the held-out targets (positions toward them come from eval episodes)
    val_targets: list[int]
    train_targets: list[int]


def make_splits(graph: WikiGraph, n_val: int = 200, seed: int = 0) -> Splits:
    held = heldout_pairs(graph)
    banned = {p for pair in held for p in pair}
    test = sorted({t for _, t in held})
    # usable targets: reachable from at least a handful of pages (29 of 300 random pages have no in-links)
    indeg = [0] * len(graph.adj)
    for u, vs in enumerate(graph.adj):
        for v in vs:
            indeg[v] += 1
    pool = [t for t in range(len(graph.adj)) if t not in banned and indeg[t] > 0]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return Splits(banned, test, sorted(pool[:n_val]), sorted(pool[n_val:]))


@dataclass
class Position:
    """A page `page` at distance `dist` from `target`, and the links of the page that the model is asked about."""
    target: int
    page: int
    dist: int
    links: list[int]
    link_dists: list[int]


class DistCache:
    def __init__(self, graph: WikiGraph, size: int = 1024):
        self.graph, self.size, self.d = graph, size, {}

    def __call__(self, target: int) -> dict[int, int]:
        if target not in self.d:
            if len(self.d) >= self.size:
                self.d.pop(next(iter(self.d)))
            self.d[target] = self.graph.dist_to(target)
        return self.d[target]


def sample_positions(graph: WikiGraph, targets: list[int], n: int, *, dists=(2, 3, 4), banned: set[int] = frozenset(),
                     max_negatives: int | None = 6, seed: int = 0, cache: DistCache | None = None) -> list[Position]:
    """`n` positions: a target, a page `dist` clicks from it (`dist` drawn evenly from `dists`; the largest includes
    farther pages), and its links that can reach the target. Training keeps every closer link plus up to
    `max_negatives` others (closer links are ~5% at 2 clicks); `max_negatives=None` keeps all links (ranking eval).
    Banned pages never appear as target, page or link."""
    rng = random.Random(seed)
    cache = cache or DistCache(graph)
    top = max(dists)
    out, tries = [], 0
    while len(out) < n and tries < n * 50:
        tries += 1
        t = rng.choice(targets)
        d = cache(t)
        k = rng.choice(dists)
        pages = [p for p, dp in d.items() if (dp == k if k < top else dp >= k) and p not in banned]
        if not pages:
            continue
        p = rng.choice(pages)
        links = [x for x in graph.adj[p] if x in d and x not in banned]  # pages >= 2 clicks out never link to t
        if not links:
            continue
        closer = [x for x in links if d[x] < d[p]]
        if not closer:
            continue
        others = [x for x in links if d[x] >= d[p]]
        if max_negatives is not None:
            others = rng.sample(others, min(max_negatives, len(others)))
        chosen = closer + others
        out.append(Position(t, p, d[p], chosen, [d[x] for x in chosen]))
    return out


def positions_from_episodes(graph: WikiGraph, episodes: list[dict], per_dist: int = 150, dists=(2, 3, 4),
                            seed: int = 0) -> list[Position]:
    """Decision positions from eval episodes (the same 450 as the 2026-11-15 probes with the defaults), every
    unvisited link that can reach the target included."""
    from .probe import collect_states, sample_states

    states = sample_states(collect_states(graph, episodes), list(dists), per_dist, seed)
    cache = DistCache(graph)
    out = []
    for s in states:
        t = graph.id_of(s["target"])
        ids = [graph.id_of(x) for x in s["path"]]
        d = cache(t)
        links = [x for x in graph.adj[ids[-1]] if x not in set(ids) and x in d]
        if links:
            out.append(Position(t, ids[-1], d[ids[-1]], links, [d[x] for x in links]))
    return out


def flatten(positions: list[Position]) -> list[tuple[int, int, int]]:
    """(page, target, class index) questions, one per link."""
    return [(x, pos.target, label(dx)) for pos in positions for x, dx in zip(pos.links, pos.link_dists)]


# ---- model side ---------------------------------------------------------------------------------------------
def class_logits(learner, prompts: list[list[int]], class_ids: list[int]) -> torch.Tensor:
    """(B, 5) logits of the five digit tokens at the answer position (prompts left-padded, no completion)."""
    pad = learner.tok.pad_token_id
    width = max(map(len, prompts))
    ids = torch.tensor([[pad] * (width - len(p)) + list(p) for p in prompts], device=learner.device)
    att = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=learner.device)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    with learner.autocast:
        logits = learner.model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=1,
                               use_cache=False).logits[:, -1]
    return logits[:, class_ids].float()


def value_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy over the five classes and the batch accuracy."""
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss, (logits.argmax(-1) == labels).float().mean()


def expected_distance(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(-1)
    return (probs * torch.arange(1, len(CLASSES) + 1, device=probs.device, dtype=probs.dtype)).sum(-1)


@torch.no_grad()
def score_positions(learner, tok, graph: WikiGraph, positions: list[Position], batch: int = 128,
                    cache: dict | None = None) -> list[list[tuple[float, int]]]:
    """Per position, per link: (expected distance, predicted class index). `cache` maps (page, target) -> result."""
    class_ids = class_token_ids(tok)
    cache = {} if cache is None else cache
    todo = sorted({(x, pos.target) for pos in positions for x in pos.links} - set(cache))
    was_training = learner.model.training
    learner.model.eval()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        prompts = [tok(prompt_text(tok, graph, x, t), add_special_tokens=False).input_ids for x, t in chunk]
        lg = class_logits(learner, prompts, class_ids)
        for key, e, c in zip(chunk, expected_distance(lg).tolist(), lg.argmax(-1).tolist()):
            cache[key] = (e, c)
    if was_training:
        learner.model.train()
    return [[cache[(x, pos.target)] for x in pos.links] for pos in positions]


def ranking_metrics(positions: list[Position], scores: list[list[tuple[float, int]]], top: int = 4) -> dict[str, float]:
    """By the position's distance (2, 3, `top`+): top-1 closer rate (the lowest expected distance is a closer link;
    ties split evenly), pairwise AUC (a closer link scores below a non-closer one on the same page), random-link
    closer rate, and classification accuracy over all asked links."""
    out: dict[str, float] = {}
    groups: dict[int, list[tuple[Position, list]]] = {}
    for pos, sc in zip(positions, scores):
        groups.setdefault(min(pos.dist, top), []).append((pos, sc))
    acc_n = acc_k = 0
    for k, items in sorted(groups.items()):
        top1, auc_num, auc_den, chance = 0.0, 0.0, 0, 0.0
        for pos, sc in items:
            e = [s[0] for s in sc]
            closer = [dx < pos.dist for dx in pos.link_dists]
            best = min(e)
            tied = [c for v, c in zip(e, closer) if math.isclose(v, best, abs_tol=1e-9)]
            top1 += sum(tied) / len(tied)
            chance += sum(closer) / len(closer)
            pos_e = [v for v, c in zip(e, closer) if c]
            neg_e = [v for v, c in zip(e, closer) if not c]
            for a in pos_e:
                for b in neg_e:
                    auc_num += 1.0 if a < b else 0.5 if a == b else 0.0
                    auc_den += 1
            for (_, cls), dx in zip(sc, pos.link_dists):
                acc_k += cls == label(dx)
                acc_n += 1
        n = len(items)
        out[f"top1_closer@{k}"] = top1 / n
        out[f"random_closer@{k}"] = chance / n
        out[f"auc@{k}"] = auc_num / auc_den if auc_den else float("nan")
        out[f"positions@{k}"] = n
    out["accuracy"] = acc_k / acc_n if acc_n else float("nan")
    return out
