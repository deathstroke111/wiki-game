"""The model being trained: a LoRA on a causal LM, per-token log-probabilities, the optimizer and checkpoints.

Sequences are laid out like TRL's: prompts left-padded to a common width, completions right-padded, explicit
position ids (so padding does not shift positions), and only the completion positions' logits are computed
(``logits_to_keep``). Log-probs are taken from logits / temperature, the distribution the answers were sampled from.
"""
from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch

LORA_DEFAULTS = dict(lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")


class Learner:
    def __init__(self, model_id: str, *, adapter: str | None = None, lora_r: int = 16, lr: float = 1e-5,
                 total_updates: int = 1, lr_schedule: str = "linear", warmup_updates: int = 0,
                 temperature: float = 1.0, dtype: torch.dtype | None = None, device: str | None = None,
                 grad_checkpointing: bool = True, max_grad_norm: float = 1.0, weight_decay: float = 0.0,
                 model=None, tok=None):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            from ..llm import pick_device
            device = pick_device()
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tok = tok or AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = model if model is not None else AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        if adapter:
            self.model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
        else:
            self.model = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, **LORA_DEFAULTS))
        self.model.to(self.device)
        if grad_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self.model.train()
        self.temperature = temperature
        self.max_grad_norm = max_grad_norm
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" and dtype == torch.bfloat16 \
            else nullcontext()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                     fused=self.device.type == "cuda")
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, _schedule(lr_schedule, total_updates, warmup_updates))
        self.updates = 0

    # ---- log-probs ---------------------------------------------------------------------------------------------
    def batch_tensors(self, prompts: list[list[int]], completions: list[list[int]]):
        pad = self.tok.pad_token_id
        pw, cw = max(map(len, prompts)), max(1, max(map(len, completions)))
        ids, att, cmask = [], [], []
        for p, c in zip(prompts, completions):
            ids.append([pad] * (pw - len(p)) + list(p) + list(c) + [pad] * (cw - len(c)))
            att.append([0] * (pw - len(p)) + [1] * (len(p) + len(c)) + [0] * (cw - len(c)))
            cmask.append([1] * len(c) + [0] * (cw - len(c)))
        t = lambda x, dt=torch.long: torch.tensor(x, dtype=dt, device=self.device)
        return t(ids), t(att), t(cmask, torch.float32), cw

    def token_logprobs(self, prompts: list[list[int]], completions: list[list[int]],
                       with_entropy: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(B, T) log-prob of each completion token, the (B, T) completion mask, and optionally the (B, T) entropy.
        Gradients flow unless called under torch.no_grad()."""
        ids, att, mask, cw = self.batch_tensors(prompts, completions)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        with self.autocast:
            logits = self.model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=cw + 1,
                                use_cache=False).logits
        logits = logits[:, :-1]  # position j predicts completion token j (model dtype, no fp32 copy kept)
        targets = ids[:, -cw:]
        logp, ent = selective_logprobs(logits, targets, 1.0 / self.temperature, with_entropy)
        return logp, mask, ent

    # ---- optimisation ------------------------------------------------------------------------------------------
    def optimizer_step(self) -> dict[str, float]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm) if self.max_grad_norm else torch.tensor(0.0)
        self.opt.step()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        self.updates += 1
        return {"grad_norm": float(norm), "lr": self.sched.get_last_lr()[0]}

    # ---- saving ------------------------------------------------------------------------------------------------
    def save_adapter(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tok.save_pretrained(str(path))
        return path

    def save_checkpoint(self, path: str | Path, state: dict) -> Path:
        """Adapter (loadable by --adapter / --resume-adapter) plus optimizer + scheduler state and trainer state."""
        path = self.save_adapter(path)
        torch.save(self.opt.state_dict(), path / "optimizer.pt")
        torch.save(self.sched.state_dict(), path / "scheduler.pt")
        (path / "trainer_state.json").write_text(json.dumps({**state, "updates": self.updates}, indent=2, default=str))
        return path

    def load_optimizer(self, path: str | Path) -> None:
        path = Path(path)
        self.opt.load_state_dict(torch.load(path / "optimizer.pt", map_location=self.device))
        self.sched.load_state_dict(torch.load(path / "scheduler.pt"))
        self.updates = json.loads((path / "trainer_state.json").read_text()).get("updates", 0)


class _SelectiveLogProb(torch.autograd.Function):
    """log softmax(logits * s)[target] per position, without keeping an fp32 copy of the (B, T, vocab) logits for
    backward: only the model-dtype logits and the per-position log-normaliser are saved, and the softmax is
    recomputed row by row in backward (d logp / d logits = s * (onehot - softmax)). With a 152k vocabulary an fp32
    copy is ~2.5 GB for 8 x 512 tokens, which ran the 24 GB card out of memory next to vLLM (2026-11-12)."""

    @staticmethod
    def forward(ctx, logits, targets, scale):
        lz = torch.empty(targets.shape, dtype=torch.float32, device=logits.device)
        lp = torch.empty_like(lz)
        for i in range(logits.shape[0]):  # one row at a time: the fp32 temporary is (T, vocab), not (B, T, vocab)
            row = logits[i].float() * scale
            lz[i] = torch.logsumexp(row, dim=-1)
            lp[i] = row.gather(-1, targets[i].unsqueeze(-1)).squeeze(-1) - lz[i]
        ctx.save_for_backward(logits, targets, lz)
        ctx.scale = scale
        return lp

    @staticmethod
    def backward(ctx, grad_lp):
        logits, targets, lz = ctx.saved_tensors
        grad = torch.empty_like(logits)
        for i in range(logits.shape[0]):
            p = torch.exp(logits[i].float() * ctx.scale - lz[i].unsqueeze(-1))
            p.neg_()
            p.scatter_add_(-1, targets[i].unsqueeze(-1), torch.ones_like(lz[i]).unsqueeze(-1))
            grad[i] = (p * (grad_lp[i].unsqueeze(-1) * ctx.scale)).to(logits.dtype)
        return grad, None, None


def selective_logprobs(logits: torch.Tensor, targets: torch.Tensor, scale: float = 1.0,
                       with_entropy: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(B, T) fp32 log-probs of ``targets`` under softmax(logits * scale), and optionally the (B, T) entropy."""
    lp = _SelectiveLogProb.apply(logits, targets, scale)
    ent = None
    if with_entropy:
        with torch.no_grad():
            ent = torch.empty_like(lp)
            for i in range(logits.shape[0]):
                row = logits[i].float() * scale
                ent[i] = torch.logsumexp(row, -1) - (torch.softmax(row, -1) * row).sum(-1)
    return lp, ent


def _schedule(kind: str, total: int, warmup: int):
    total = max(1, total)

    def f(i: int) -> float:
        if warmup and i < warmup:
            return (i + 1) / warmup
        if kind == "constant":
            return 1.0
        frac = (i - warmup) / max(1, total - warmup)
        if kind == "cosine":
            return 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))
        return max(0.0, 1.0 - frac)  # linear decay to 0, the TRL default every stage so far used
    if kind not in ("linear", "constant", "cosine"):
        raise ValueError(f"unknown lr schedule {kind!r}")
    return f
