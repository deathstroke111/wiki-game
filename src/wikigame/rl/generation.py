"""Text generation behind one interface, so the trainer, on-policy rounds and evals do not care what samples.

  VLLMGenerator  vLLM in the same process (Linux + CUDA, the ``vllm`` dependency group). The LoRA being trained is
                 handed over as an adapter directory after each update (``set_adapter``); vLLM loads it as a new
                 LoRA (a fresh id each time), which is the path the stage-3 smoke test exercised.
  HFGenerator    ``model.generate`` on the model being trained (the Mac, tests, or no vLLM). It always samples from
                 the live weights, so ``set_adapter`` is a no-op.
  ScriptedGenerator  canned answers for tests (no model).

Prompts go in as token ids, so the trainer scores exactly the tokens the sampler saw (no re-tokenisation drift).
``chat_batch_generate`` wraps any generator as the ``play_games`` model hook (chat messages in, texts out).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass
class Sample:
    token_ids: list[int]  # completion tokens, including the end-of-sequence token when the answer finished
    text: str  # decoded completion, special tokens removed
    finished: bool  # stopped on its own (False = cut off at max_tokens)
    logprobs: list[float] | None = None  # sampler's log-prob of each completion token (vLLM only)


class Generator(Protocol):
    def generate(self, prompt_ids: list[list[int]], n: int, max_tokens: int, temperature: float,
                 seed: int | None = None) -> list[list[Sample]]: ...

    def set_adapter(self, path: str | None) -> None: ...


@dataclass
class GenStats:
    calls: int = 0  # prompts answered (n answers each)
    answers: int = 0
    prompt_tokens: int = 0
    new_tokens: int = 0
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def add(self, prompt_ids: list[list[int]], out: list[list[Sample]], seconds: float) -> None:
        self.calls += len(prompt_ids)
        self.answers += sum(len(o) for o in out)
        self.prompt_tokens += sum(map(len, prompt_ids))
        self.new_tokens += sum(len(s.token_ids) for o in out for s in o)
        self.seconds += seconds

    def summary(self) -> dict:
        n = max(1, self.answers)
        return {"calls": self.answers, "avg_prompt_tokens": self.prompt_tokens / max(1, self.calls),
                "avg_new_tokens": self.new_tokens / n, "sec_per_call": self.seconds / n}


class VLLMGenerator:
    """vLLM engine with LoRA enabled. ``gpu_memory_utilization`` is this engine's share of the card (weights + KV
    cache); the trainer uses the rest. ``logprobs_mode='processed_logprobs'`` returns log-probs after temperature,
    the same distribution the trainer scores (logits / T)."""

    wants_adapter_files = True  # the trainer saves the LoRA after each update and calls set_adapter with its dir

    def __init__(self, model_id: str, *, gpu_memory_utilization: float = 0.35, max_model_len: int = 4096,
                 max_lora_rank: int = 16, seed: int = 0, dtype: str = "bfloat16", enable_sleep_mode: bool = False,
                 adapter: str | None = None):
        from vllm import LLM

        kw = dict(model=model_id, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization,
                  max_model_len=max_model_len, enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1, seed=seed,
                  enable_sleep_mode=enable_sleep_mode, logprobs_mode="processed_logprobs")
        t0 = time.time()
        try:
            self.llm = LLM(**kw)
        except TypeError:  # older vLLM without logprobs_mode: raw log-probs (before temperature); only diagnostics
            kw.pop("logprobs_mode")
            self.llm = LLM(**kw)
        self.startup_seconds = time.time() - t0
        self.sleeping = False
        self._lora = None
        self._next_id = 1
        self.stats = GenStats()
        if adapter:
            self.set_adapter(adapter)

    def set_adapter(self, path: str | None) -> None:
        from vllm.lora.request import LoRARequest

        old = self._lora
        self._lora = LoRARequest(f"adapter-{self._next_id}", self._next_id, str(path)) if path else None
        self._next_id += 1
        if old is not None:  # free the previous adapter's slot (max_loras=1); the next request loads the new one
            try:
                self.llm.llm_engine.remove_lora(old.lora_int_id)
            except Exception:  # noqa: BLE001 - API moved between vLLM versions; an LRU eviction does the same
                pass

    def sleep(self) -> None:
        if not self.sleeping:
            self.llm.sleep(level=1)
            self.sleeping = True

    def wake_up(self) -> None:
        if self.sleeping:
            self.llm.wake_up()
            self.sleeping = False

    def generate(self, prompt_ids, n, max_tokens, temperature, seed=None):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        self.wake_up()
        sp = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, logprobs=0, seed=seed)
        t0 = time.time()
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=list(p)) for p in prompt_ids], sp,
                                 lora_request=self._lora, use_tqdm=False)
        res = []
        for o in outs:
            group = []
            for c in o.outputs:
                ids = list(c.token_ids)
                lps = None
                if c.logprobs:
                    lps = [float(step[t].logprob) for step, t in zip(c.logprobs, ids)]
                group.append(Sample(ids, c.text, c.finish_reason == "stop", lps))
            res.append(group)
        self.stats.add(prompt_ids, res, time.time() - t0)
        return res


class HFGenerator:
    """Batched ``model.generate`` on token ids (left-padded). Samples the model as it is right now."""

    def __init__(self, model, tok, device=None, max_batch: int = 64):
        self.model, self.tok, self.max_batch = model, tok, max_batch
        self.device = device
        self.stats = GenStats()
        self.eos = sorted({i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None})

    def set_adapter(self, path: str | None) -> None:
        pass  # generates with the live (training) weights

    def generate(self, prompt_ids, n, max_tokens, temperature, seed=None):
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        flat = [p for p in prompt_ids for _ in range(n)]
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        device = self.device or next(self.model.parameters()).device
        was_training = self.model.training
        ckpt = bool(getattr(self.model, "is_gradient_checkpointing", False))
        if ckpt:  # checkpointing forces use_cache=False (every new token re-reads the whole prompt); off while sampling
            self.model.gradient_checkpointing_disable()
        self.model.eval()
        t0 = time.time()
        texts: list[Sample] = []
        try:
            for i in range(0, len(flat), self.max_batch):
                chunk = flat[i:i + self.max_batch]
                width = max(map(len, chunk))
                ids = torch.tensor([[pad] * (width - len(p)) + list(p) for p in chunk], device=device)
                att = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in chunk], device=device)
                kw = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
                with torch.inference_mode():
                    out = self.model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_tokens,
                                              pad_token_id=pad, use_cache=True, **kw)
                for row in out[:, width:].tolist():
                    stop = next((j for j, t in enumerate(row) if t in self.eos), None)
                    toks = row[:stop + 1] if stop is not None else row[:max_tokens]
                    texts.append(Sample(toks, self.tok.decode(toks, skip_special_tokens=True), stop is not None))
        finally:
            if ckpt:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.train(was_training)
        res = [texts[i * n:(i + 1) * n] for i in range(len(prompt_ids))]
        self.stats.add(prompt_ids, res, time.time() - t0)
        return res


class ScriptedGenerator:
    """Test double: ``answer(prompt_ids, k) -> text`` for the k-th answer to a prompt; tokenised with ``tok``."""

    def __init__(self, tok, answer: Callable[[list[int], int], str]):
        self.tok, self.answer = tok, answer
        self.stats = GenStats()
        self.adapters: list[str | None] = []

    def set_adapter(self, path: str | None) -> None:
        self.adapters.append(path)

    def generate(self, prompt_ids, n, max_tokens, temperature, seed=None):
        res = []
        for p in prompt_ids:
            group = []
            for k in range(n):
                text = self.answer(p, k)
                ids = self.tok(text, add_special_tokens=False).input_ids[:max_tokens]
                finished = len(ids) < max_tokens
                if finished:
                    ids = ids + [self.tok.eos_token_id]
                group.append(Sample(ids, text, finished))
            res.append(group)
        self.stats.add(prompt_ids, res, 0.0)
        return res


def encode_prompt(tok, messages: list[dict]) -> tuple[list[int], bool]:
    """Chat-template a prompt the way every script does (template text, then no extra special tokens). Returns the
    token ids and whether the template opened a <think> block (R1), which the answer text must get back."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids, text.rstrip().endswith("<think>")


def restore_think(text: str, opened: bool) -> str:
    return "<think>\n" + text if opened else text


def chat_batch_generate(gen: Generator, tok, max_tokens: int, temperature: float,
                        seed: int | None = None) -> Callable[[list[list[dict]]], list[str]]:
    """``play_games``-style hook: a batch of chat messages -> one answer text each (with <think> restored)."""
    def generate(batch: list[list[dict]]) -> list[str]:
        enc = [encode_prompt(tok, m) for m in batch]
        out = gen.generate([ids for ids, _ in enc], 1, max_tokens, temperature, seed)
        return [restore_think(o[0].text, opened) for o, (_, opened) in zip(out, enc)]
    return generate
