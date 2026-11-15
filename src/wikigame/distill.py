"""Value step 2: distil the value's top link into the single-answer policy (the user's design, 2026-11-16).

Data (one row per game position, toward TRAINING targets only, no held-out page on the path):
  - the policy's normal prompt; 8 of the policy's own answers sampled there;
  - the label: the frozen step-0 value's top link (the target itself when it is linked, as in value play);
  - the reasoning kept: a sample whose own answer already is the label (reasoning and answer agree) if any,
    else the first finished sample's reasoning with its answer swapped for the label. Rows whose samples were all
    cut off before ``</think>`` are dropped.
Training (``scripts/distill_train.py``): the completion is ``reasoning + <answer>{label}</answer><eos>``; the loss
covers the answer tokens only (the model's own reasoning is context, not a target), mixed 50/50 with value questions.
"""
from __future__ import annotations

import json
import random

from .env import parse_action
from .graph import WikiGraph
from .llm import lenient_answer
from .states import StateMix, sample_states
from .value import Splits

THINK_END = "</think>"


def strict_states(graph: WikiGraph, n: int, model_id: str, splits: Splits, targets: set[int], seed: int = 0,
                  mix: StateMix | None = None) -> list[dict]:
    """Up to `n` policy training states whose target is in `targets` and whose path has no banned page."""
    mix = mix or StateMix(hops=(2, 3, 4))
    rows = sample_states(graph, 4 * n, model_id, mix, seed=seed)
    keep = [r for r in rows if r["target"] in targets and not any(p in splits.banned for p in r["path"])]
    return keep[:n]


def sample_answer(raw: str) -> str | None:
    """The link a raw sample (the text after the template's '<think>\\n') answers, leniently parsed."""
    return parse_action(lenient_answer("<think>\n" + raw))


def answer_text(title: str) -> str:
    return f"<answer>{json.dumps({'title': title})}</answer>"


def choose_reasoning(samples: list[str], label: str) -> tuple[str, bool] | None:
    """(reasoning prefix ending in '</think>\\n\\n', matched) from raw samples, or None if none finished thinking.
    matched = the sample's own answer was already the label."""
    finished = [s for s in samples if THINK_END in s]
    if not finished:
        return None
    match = next((s for s in finished if sample_answer(s) == label), None)
    s = match if match is not None else finished[0]
    return s[: s.index(THINK_END) + len(THINK_END)] + "\n\n", match is not None


def label_for(graph: WikiGraph, target: int, links: list[str], scores: list[float], rng: random.Random) -> str:
    """Value play's choice: the target if linked, else the lowest expected distance (ties at random)."""
    from .value import value_choice

    return value_choice(graph, target, links, scores, rng)[0]


def move_example(tok, prompt_ids: list[int], reasoning: str, label: str) -> tuple[list[int], list[int], int]:
    """(prompt ids, completion ids, answer start): completion = reasoning + answer + eos; loss from answer start."""
    r = tok(reasoning, add_special_tokens=False).input_ids
    a = tok(answer_text(label), add_special_tokens=False).input_ids + [tok.eos_token_id]
    return list(prompt_ids), r + a, len(r)
