"""Prompted-LLM policy (zero-shot baseline now; the same prompt format is what GRPO will train).

Usage: ``policy = HFPolicy("Qwen/Qwen2.5-1.5B-Instruct")`` then ``run_episode(env, policy)``.
Greedy decoding by default so a baseline run is deterministic. Never uses trust_remote_code.
"""
from __future__ import annotations

import json
import os
import re
import time

from .env import SYSTEM_PROMPT, WikiGameEnv, parse_action
from .graph import WikiGraph

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_messages(obs: str) -> list[dict]:
    """Chat messages for one step. The observation is self-contained (Markov), so there is no history."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}]


_BARE_TITLE = re.compile(r'\{[^{}]*"title"\s*:\s*("(?:[^"\\]|\\.)*")[^{}]*\}')


def lenient_answer(text: str) -> str:
    """If the model gave a title but not in our tag format (bare or ```json-fenced JSON), append a canonical
    <answer> so the env can parse it. The raw text is kept. Well-formed or title-less output is unchanged.

    Used only for the 'lenient' baseline variant, to separate format failures from navigation failures."""
    if parse_action(text) is not None:
        return text
    m = _BARE_TITLE.search(text)
    if not m:
        return text
    try:
        title = json.loads(m.group(1))
    except json.JSONDecodeError:
        return text
    return f"{text}\n<answer>{json.dumps({'title': title})}</answer>"


# R1 distills otherwise reason about every one of the ~200 links and run out of tokens before answering.
R1_HINT = ("Reason briefly (a few sentences). Do NOT go through the links one by one: name the one or two most "
           "promising links for reaching the target, then answer immediately with the <answer> tag.")


# Stage-4 spike: wordings that ask for a little more reasoning (stage 3 used ~132 of its 512 tokens). All keep the
# "not one by one" guard, which is what stopped the cut-offs. "brief" is the wording every run so far used.
HINTS = {
    "brief": R1_HINT,
    "bridge": ("Think in two short steps. First say what the target is about and which broader topic (a country, "
               "field, era or category) connects the current article to it. Then pick the link closest to that topic. "
               "Do NOT go through the links one by one. Then answer with the <answer> tag."),
    "compare": ("Do NOT go through the links one by one. Pick the three most promising links and say in one sentence "
                "each what that article probably links to. Then choose the one most likely to lead to the target and "
                "answer with the <answer> tag."),
    "plan": ("Do NOT go through the links one by one. Plan two moves ahead: name the link you would click now and the "
             "link you would expect to find and click on that next page. If no link is clearly related to the target, "
             "prefer a broad, well-connected article. Then answer with the <answer> tag."),
}


def resolve_hint(hint: str | None) -> str | None:
    """A HINTS name -> its text; any other string is used as the hint itself ("" disables, None = model default)."""
    return HINTS.get(hint, hint) if hint else hint


def is_r1(model_id: str) -> bool:
    return "deepseek-r1" in model_id.lower()


def merge_system(messages: list[dict], hint: str | None = None) -> list[dict]:
    """Fold the system message (plus an optional extra hint) into the first user message
    (R1-distill recommended prompting)."""
    sys_ = [m["content"] for m in messages if m["role"] == "system"]
    if sys_ and hint:
        sys_[-1] = f"{sys_[-1]} {hint}"
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if sys_ and rest and rest[0]["role"] == "user":
        rest[0]["content"] = "\n\n".join(sys_ + [rest[0]["content"]])
    return rest


def training_messages(obs: str, model_id: str, hint: str | None = None) -> list[dict]:
    """Chat messages exactly as HFPolicy.generate will send them for ``model_id``, so GRPO trains on the same
    prompt format the zero-shot baselines and evals use (R1: system prompt + brevity hint folded into the user turn)."""
    msgs = build_messages(obs)
    if is_r1(model_id):
        msgs = merge_system(msgs, R1_HINT if hint is None else (hint or None))
    return msgs


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batch_generate(model, tok, texts: list[str], max_new_tokens: int, temperature: float,
                   device=None) -> tuple[list[str], list[int], int]:
    """One left-padded ``generate`` call over chat-templated ``texts``. Returns (decoded answers, new tokens per
    answer up to and including its end token, total prompt tokens). Restores the opening <think> tag that R1's
    template puts in the prompt, like HFPolicy.generate."""
    import torch

    side = tok.padding_side
    tok.padding_side = "left"
    try:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        tok.padding_side = side
    enc = enc.to(device or model.device)
    kw = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, use_cache=True, **kw)
    new = out[:, enc["input_ids"].shape[1]:].cpu()
    eos = torch.tensor(sorted({tok.eos_token_id, tok.pad_token_id}))
    answers, n_new = [], []
    for text, row in zip(texts, new):
        stop = torch.isin(row, eos).nonzero()
        n = int(stop[0]) + 1 if len(stop) else len(row)
        out_text = tok.decode(row[:n], skip_special_tokens=True)
        if text.rstrip().endswith("<think>"):
            out_text = "<think>\n" + out_text
        answers.append(out_text)
        n_new.append(n)
    return answers, n_new, int(enc["attention_mask"].sum())


class HFPolicy:
    """policy(obs, env) -> raw model text. Tracks generation stats for reporting."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None, max_new_tokens: int = 256,
                 temperature: float = 0.0, dtype: str = "bfloat16", lenient: bool = False,
                 system_in_user: bool | None = None, seed: int = 0, hint: str | None = None,
                 adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id, self.max_new_tokens, self.temperature = model_id, max_new_tokens, temperature
        self.lenient = lenient
        # DeepSeek-R1 distills are meant to be prompted without a system prompt (instructions go in the user turn)
        self.system_in_user = is_r1(model_id) if system_in_user is None else system_in_user
        self.hint = (R1_HINT if is_r1(model_id) else None) if hint is None else (hint or None)  # "" disables
        torch.manual_seed(seed)  # only matters when temperature > 0
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(self.device)
        if adapter:  # a LoRA adapter from GRPO training, applied UNMERGED
            # merge_and_unload() into bf16 weights rounded most of the update away (2026-11-12: ~80% of weights
            # unchanged, ~half of the update kept), so every HF eval of a trained adapter before then under-measured it
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model = self.model.eval()
        self.adapter = adapter
        self.calls = self.prompt_tokens = self.new_tokens = 0
        self.seconds = 0.0

    def generate(self, messages: list[dict]) -> str:
        import torch

        if self.system_in_user:
            messages = merge_system(messages, self.hint)
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        kw = {"do_sample": True, "temperature": self.temperature} if self.temperature > 0 else {"do_sample": False}
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      pad_token_id=self.tok.eos_token_id, **kw)
        new = out[0, inputs["input_ids"].shape[1]:]
        self.calls += 1
        self.prompt_tokens += inputs["input_ids"].shape[1]
        self.new_tokens += len(new)
        self.seconds += time.time() - t0
        text_out = self.tok.decode(new, skip_special_tokens=True)
        if text.rstrip().endswith("<think>"):  # R1's template opens the think block in the prompt; restore it
            text_out = "<think>\n" + text_out
        return lenient_answer(text_out) if self.lenient else text_out

    def generate_batch(self, batch: list[list[dict]]) -> list[str]:
        """Like ``generate`` for several prompts in one padded call (the parallel game player's model hook).
        Sampling noise differs from one-at-a-time calls, so batched and sequential evals are not sample-identical."""
        if self.system_in_user:
            batch = [merge_system(m, self.hint) for m in batch]
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        t0 = time.time()
        outs, n_new, n_prompt = batch_generate(self.model, self.tok, texts, self.max_new_tokens, self.temperature,
                                               self.device)
        self.calls += len(texts)
        self.prompt_tokens += n_prompt
        self.new_tokens += sum(n_new)
        self.seconds += time.time() - t0  # wall time of the batch: sec_per_call becomes time per batched answer
        return [lenient_answer(o) for o in outs] if self.lenient else outs

    def __call__(self, obs: str, env: WikiGameEnv) -> str:
        return self.generate(build_messages(obs))

    def stats(self) -> dict:
        n = max(1, self.calls)
        return {"calls": self.calls, "avg_prompt_tokens": self.prompt_tokens / n,
                "avg_new_tokens": self.new_tokens / n, "sec_per_call": self.seconds / n}


def prompt_token_stats(graph: WikiGraph, tokenizer, n: int = 20, seed: int = 0, max_links: int = 200) -> dict:
    """How long are our prompts? Samples start states and measures chat-template token counts."""
    import random

    rng = random.Random(seed)
    env = WikiGameEnv(graph, max_links_shown=max_links, seed=seed)
    lens = []
    for _ in range(n):
        s, t = graph.sample_pair(rng.choice((1, 2, 3)), rng)
        obs, _ = env.reset(options={"source": s, "target": t})
        text = tokenizer.apply_chat_template(build_messages(obs), tokenize=False, add_generation_prompt=True)
        lens.append(len(tokenizer(text)["input_ids"]))
    lens.sort()
    return {"min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1]}


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
