"""Step 0 of the value plan: supervised "how many clicks to the target?" training, and the go/no-go measurement.

  1. Baseline: score positions with the starting adapter (validation, test, train-target check).
  2. Train the value task (cross-entropy over the digits 1-5) on training targets only; every --eval-every steps score
     the validation positions and keep the adapter with the best top-1 closer rate at 2 clicks.
  3. Score the test positions (decision positions toward the held-out targets, from an eval's episodes) and the
     train-target check with the best adapter. The test set is only scored before training and once at the end.

Decision rule (fixed in advance, 2026-11-15): go on to value-ranked play if test top1_closer@2 >= 0.35, stop if
<= 0.30, otherwise one retry (rank 64 or twice the data).

Pod:
  uv run python scripts/value_train.py --adapter data/ckpt/grpo-step-r1-stage4 \
      --episodes data/runs/grpo-step-r1-stage4-final-vllm-cuda-1794506767.jsonl --out data/ckpt/value-step0
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from wikigame import WikiGraph
from wikigame.value import (DistCache, class_logits, class_token_ids, flatten, make_splits, positions_from_episodes,
                            prompt_text, ranking_metrics, sample_positions, score_positions, value_loss)

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", default=None, help="starting LoRA adapter (default: a fresh rank --lora-r LoRA)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--episodes", required=True, help="eval episodes whose positions (toward held-out targets) are the test set")
    ap.add_argument("--graph", default="data/wikispeedia.json")
    ap.add_argument("--graph-name", default="wikispeedia")
    ap.add_argument("--out", default="data/ckpt/value-step0")
    ap.add_argument("--train-positions", type=int, default=20000)
    ap.add_argument("--val-positions", type=int, default=900)
    ap.add_argument("--check-positions", type=int, default=450, help="positions toward TRAINING targets (memorisation check)")
    ap.add_argument("--test-per-dist", type=int, default=150)
    ap.add_argument("--max-negatives", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="trade speed for memory; measured on the RTX PRO 4000 (2026-11-15): off + micro 8 = 3.5 s per "
                         "128 questions at 10 GB, on + micro 32 = 5.4 s at 6 GB")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="cap on optimizer steps (smoke tests)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="score the test and train-target check positions with --adapter and exit (no training, no "
                         "W&B); e.g. to see whether a step-2 adapter kept the value")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    from wikigame.rl.learner import Learner

    graph = WikiGraph.load(a.graph)
    splits = make_splits(graph, seed=a.seed)
    cache = DistCache(graph)
    t0 = time.time()
    train_pos = sample_positions(graph, splits.train_targets, a.train_positions, banned=splits.banned,
                                 max_negatives=a.max_negatives, seed=a.seed, cache=cache)
    val_pos = sample_positions(graph, splits.val_targets, a.val_positions, banned=splits.banned, max_negatives=None,
                               seed=a.seed + 1, cache=cache)
    check_pos = sample_positions(graph, splits.train_targets, a.check_positions, banned=splits.banned,
                                 max_negatives=None, seed=a.seed + 2, cache=cache)
    episodes = [json.loads(line) for line in Path(a.episodes).read_text().splitlines() if line]
    test_pos = positions_from_episodes(graph, episodes, per_dist=a.test_per_dist, seed=a.seed)
    assert not {p.target for p in train_pos} & set(splits.test_targets), "held-out target in training"
    questions = flatten(train_pos)
    random.shuffle(questions)
    closer_share = sum(1 for pos in train_pos for dx in pos.link_dists if dx < pos.dist) / len(questions)
    print(f"splits: {len(splits.train_targets)} train / {len(splits.val_targets)} val / {len(splits.test_targets)} test "
          f"targets, {len(splits.banned)} banned pages | positions: {len(train_pos)} train ({len(questions)} questions, "
          f"{closer_share:.0%} closer links), {len(val_pos)} val, {len(check_pos)} check, {len(test_pos)} test "
          f"| {time.time() - t0:.0f}s", flush=True)

    steps_per_epoch = max(1, len(questions) // a.batch)
    total = int(steps_per_epoch * a.epochs)
    if a.max_steps:
        total = min(total, a.max_steps)
    L = Learner(a.model, adapter=a.adapter, lora_r=a.lora_r, lr=a.lr, total_updates=total, lr_schedule="cosine",
                warmup_updates=a.warmup, device=a.device, grad_checkpointing=a.grad_checkpointing)
    tok = L.tok
    class_ids = class_token_ids(tok)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.eval_only:
        res = {}
        for name, pos in (("test", test_pos), ("check", check_pos)):
            m = ranking_metrics(pos, score_positions(L, tok, graph, pos, batch=a.eval_batch))
            res.update({f"{name}/{k}": v for k, v in m.items()})
            print(f"[{name}] " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if not k.startswith("positions")), flush=True)
        (out / "eval-only.json").write_text(json.dumps({"adapter": a.adapter, **res}, indent=2))
        print(f"eval-only results in {out / 'eval-only.json'}")
        return

    run = None
    if not a.no_wandb:
        from wikigame.tracking import init_run

        experiment = f"value-step0-r1-from-{Path(a.adapter).name if a.adapter else 'base'}"
        run = init_run(experiment, "value-train", {**vars(a), "experiment": experiment, "n_questions": len(questions),
                                                    "closer_share": closer_share, "total_steps": total},
                       graph, a.graph_name, tags=["value", "step0"], group="value")

    def log(metrics: dict, step: int) -> None:
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if run:
            run.log(metrics, step=step)

    def evaluate(name: str, positions, step: int) -> dict:
        t = time.time()
        m = ranking_metrics(positions, score_positions(L, tok, graph, positions, batch=a.eval_batch))
        print(f"[{name} @ step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if not k.startswith("positions"))
              + f" ({time.time() - t:.0f}s)", flush=True)
        return {f"{name}/{k}": v for k, v in m.items()}

    results = {"baseline": {}, "best": {}}
    for name, pos in (("val", val_pos), ("test", test_pos), ("check", check_pos)):
        results["baseline"].update(evaluate(name, pos, 0))
    log({f"baseline/{k}": v for k, v in results["baseline"].items()}, 0)

    trainable = [p for p in L.model.parameters() if p.requires_grad]
    # the starting adapter is the fallback "best": if no validation check beats it, the final eval scores it, not the
    # last-trained weights (bug found in the 2026-11-15 Mac smoke run)
    best_score, best_step = results["baseline"]["val/top1_closer@2"], 0
    best_state = [p.detach().cpu().clone() for p in trainable]
    cursor = 0
    for step in range(1, total + 1):
        t = time.time()
        batch = [questions[(cursor + i) % len(questions)] for i in range(a.batch)]
        cursor += a.batch
        loss_sum = acc_sum = 0.0
        for i in range(0, len(batch), a.micro_batch):
            mb = batch[i:i + a.micro_batch]
            prompts = [tok(prompt_text(tok, graph, x, tg), add_special_tokens=False).input_ids for x, tg, _ in mb]
            labels = torch.tensor([c for _, _, c in mb], device=L.device)
            loss, acc = value_loss(class_logits(L, prompts, class_ids), labels)
            (loss * len(mb) / len(batch)).backward()
            loss_sum += loss.item() * len(mb)
            acc_sum += acc.item() * len(mb)
        opt = L.optimizer_step()
        m = {"train/loss": loss_sum / len(batch), "train/acc": acc_sum / len(batch), "train/lr": opt["lr"],
             "train/grad_norm": opt["grad_norm"], "train/step_s": time.time() - t}
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{total} loss {m['train/loss']:.3f} acc {m['train/acc']:.3f} lr {opt['lr']:.2e} "
                  f"gnorm {opt['grad_norm']:.2f} {m['train/step_s']:.2f}s", flush=True)
        if step % a.eval_every == 0 or step == total:
            vm = evaluate("val", val_pos, step)
            m.update(vm)
            if vm["val/top1_closer@2"] > best_score:
                best_score, best_step = vm["val/top1_closer@2"], step
                best_state = [p.detach().cpu().clone() for p in trainable]
                L.save_adapter(out / "best")
                print(f"  new best val top1_closer@2 {best_score:.3f} at step {step}", flush=True)
        log(m, step)
    L.save_adapter(out / "final")

    with torch.no_grad():
        for p, b in zip(trainable, best_state):
            p.copy_(b.to(p.device))
    results["best_step"] = best_step
    for name, pos in (("val", val_pos), ("test", test_pos), ("check", check_pos)):
        results["best"].update(evaluate(name, pos, best_step))
    log({f"best/{k}": v for k, v in results["best"].items()} | {"best/step": best_step}, total + 1)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    b, f = results["baseline"], results["best"]
    print(f"\n== step-0 result (best = step {best_step}; baseline = starting adapter) ==")
    for k in ("top1_closer@2", "top1_closer@3", "top1_closer@4", "auc@2", "auc@3", "auc@4", "random_closer@2",
              "random_closer@3", "random_closer@4", "accuracy"):
        print(f"{k:18s} test {b[f'test/{k}']:.3f} -> {f[f'test/{k}']:.3f} | train-target check {b[f'check/{k}']:.3f} -> "
              f"{f[f'check/{k}']:.3f} | val {b[f'val/{k}']:.3f} -> {f[f'val/{k}']:.3f}")
    t2 = f["test/top1_closer@2"]
    verdict = "GO (>= 0.35)" if t2 >= 0.35 else "STOP (<= 0.30)" if t2 <= 0.30 else "RETRY ZONE (0.30-0.35)"
    print(f"decision: test top1_closer@2 = {t2:.3f} -> {verdict}")
    if run:
        run.summary["decision"] = verdict
        run.finish()


if __name__ == "__main__":
    main()
