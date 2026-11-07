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


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFPolicy:
    """policy(obs, env) -> raw model text. Tracks generation stats for reporting."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None, max_new_tokens: int = 256,
                 temperature: float = 0.0, dtype: str = "bfloat16", lenient: bool = False,
                 system_in_user: bool | None = None, seed: int = 0, hint: str | None = None):
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
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(self.device).eval()
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
