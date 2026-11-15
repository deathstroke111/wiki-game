"""Value step 2, training: teach the policy to answer the value's top link, keeping the value task alive.

Every optimizer step = `--moves` move examples (prompt + the model's own reasoning + <answer>label</answer>, loss on
the answer tokens only) and `--values` value questions (5-class cross-entropy, as step 0); the two losses are averaged
with equal weight (the user's 50/50 mix). Starts from the step-0 value adapter. Validation every --eval-every steps:
answer NLL on the val rows (validation targets) and value top-1-closer on validation positions.

Pod:
  uv run python scripts/distill_train.py --adapter data/ckpt/value-step0/best --rows data/distill/step2.jsonl \
      --out data/ckpt/distill-step2
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from wikigame import WikiGraph
from wikigame.distill import move_example
from wikigame.value import (DistCache, class_logits, class_token_ids, flatten, make_splits, prompt_text,
                            ranking_metrics, sample_positions, score_positions, value_loss)

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def answer_nll(L, examples: list[tuple[list[int], list[int], int]]) -> torch.Tensor:
    """Per-example mean NLL of the answer tokens (completion positions >= answer start)."""
    lp, mask, _ = L.token_logprobs([p for p, _, _ in examples], [c for _, c, _ in examples])
    am = torch.zeros_like(mask)
    for i, (_, c, start) in enumerate(examples):
        am[i, start:len(c)] = 1.0
    return -(lp * am).sum(-1) / am.sum(-1).clamp(min=1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", required=True, help="starting adapter (the step-0 value adapter)")
    ap.add_argument("--rows", required=True, help="distill_data.py output")
    ap.add_argument("--graph", default="data/wikispeedia.json")
    ap.add_argument("--graph-name", default="wikispeedia")
    ap.add_argument("--out", default="data/ckpt/distill-step2")
    ap.add_argument("--moves", type=int, default=8, help="move examples per optimizer step")
    ap.add_argument("--values", type=int, default=8, help="value questions per optimizer step")
    ap.add_argument("--micro-moves", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the move rows")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-value-positions", type=int, default=150)
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    from wikigame.rl.learner import Learner

    graph = WikiGraph.load(a.graph)
    splits = make_splits(graph, seed=0)
    rows = [json.loads(line) for line in Path(a.rows).read_text().splitlines() if line]
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    assert not {r["target"] for r in train_rows} & set(splits.test_targets), "held-out target in the move rows"
    random.shuffle(train_rows)
    total = int(len(train_rows) * a.epochs / a.moves)
    if a.max_steps:
        total = min(total, a.max_steps)
    cache = DistCache(graph)
    vpos = sample_positions(graph, splits.train_targets, max(1000, total * a.values // 8), banned=splits.banned,
                            seed=a.seed + 10, cache=cache)
    vq = flatten(vpos)
    random.shuffle(vq)
    val_vpos = sample_positions(graph, splits.val_targets, a.val_value_positions, banned=splits.banned,
                                max_negatives=None, seed=a.seed + 1, cache=cache)
    print(f"{len(train_rows)} train move rows ({sum(r['matched'] for r in train_rows) / max(1, len(train_rows)):.0%} "
          f"reasoning already matched), {len(val_rows)} val rows, {len(vq)} value questions, {total} steps", flush=True)

    L = Learner(a.model, adapter=a.adapter, lr=a.lr, total_updates=total, lr_schedule="cosine", warmup_updates=a.warmup,
                device=a.device, grad_checkpointing=not a.no_grad_checkpointing, temperature=1.0)
    tok = L.tok
    class_ids = class_token_ids(tok)
    ex = lambda r: move_example(tok, r["prompt_ids"], r["reasoning"], r["label"])  # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if not a.no_wandb:
        from wikigame.tracking import init_run

        run = init_run("distill-step2-r1", "distill-train", {**vars(a), "train_rows": len(train_rows),
                                                             "total_steps": total}, graph, a.graph_name,
                       tags=["value", "step2", "distill"], group="value")

    def log(m: dict, step: int) -> None:
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **m}) + "\n")
        if run:
            run.log(m, step=step)

    def validate(step: int) -> dict:
        t = time.time()
        L.model.eval()
        with torch.no_grad():
            nll = [answer_nll(L, [ex(r) for r in val_rows[i:i + a.micro_moves]]) for i in range(0, len(val_rows), a.micro_moves)]
        L.model.train()
        m = {"val/answer_nll": torch.cat(nll).mean().item() if nll else float("nan")}
        vm = ranking_metrics(val_vpos, score_positions(L, tok, graph, val_vpos))
        m.update({f"val/value_{k}": v for k, v in vm.items() if not k.startswith("positions")})
        print(f"[val @ {step}] answer_nll {m['val/answer_nll']:.3f} value top1_closer@2 {vm['top1_closer@2']:.3f} "
              f"@3 {vm['top1_closer@3']:.3f} acc {vm['accuracy']:.3f} ({time.time() - t:.0f}s)", flush=True)
        return m

    log(validate(0), 0)
    mc = vc = 0
    for step in range(1, total + 1):
        t = time.time()
        moves = [train_rows[(mc + i) % len(train_rows)] for i in range(a.moves)]
        mc += a.moves
        mloss = 0.0
        for i in range(0, len(moves), a.micro_moves):
            nll = answer_nll(L, [ex(r) for r in moves[i:i + a.micro_moves]])
            (0.5 * nll.sum() / len(moves)).backward()
            mloss += nll.sum().item()
        vb = [vq[(vc + i) % len(vq)] for i in range(a.values)]
        vc += a.values
        prompts = [tok(prompt_text(tok, graph, x, tg), add_special_tokens=False).input_ids for x, tg, _ in vb]
        vloss, vacc = value_loss(class_logits(L, prompts, class_ids), torch.tensor([c for _, _, c in vb], device=L.device))
        (0.5 * vloss).backward()
        opt = L.optimizer_step()
        m = {"train/answer_nll": mloss / len(moves), "train/value_loss": vloss.item(), "train/value_acc": vacc.item(),
             "train/lr": opt["lr"], "train/grad_norm": opt["grad_norm"], "train/step_s": time.time() - t}
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{total} answer_nll {m['train/answer_nll']:.3f} value_loss {m['train/value_loss']:.3f} "
                  f"value_acc {m['train/value_acc']:.2f} lr {opt['lr']:.2e} {m['train/step_s']:.1f}s", flush=True)
        if step % a.eval_every == 0 or step == total:
            m.update(validate(step))
            L.save_adapter(out / f"checkpoint-{step}")
        log(m, step)
    L.save_adapter(out / "final")
    if run:
        run.finish()
    print("adapter saved to", out / "final")


if __name__ == "__main__":
    main()
