"""Train with our own RL loop (wikigame.rl): vLLM (or HF) generation + a pluggable policy-gradient algorithm.

Same states, reward, on-policy rounds and checkpoint format as scripts/grpo_train.py (TRL), so adapters from either
evaluate the same way:  scripts/zero_shot.py --adapter <out> [--backend vllm].

  # plumbing check on the Mac (HF generation, tiny model; numbers mean nothing)
  HF_HOME=data/hf uv run python scripts/rl_train.py --model Qwen/Qwen2.5-0.5B-Instruct --backend hf --algo grpo \
      --max-steps 2 --prompts-per-step 1 --num-generations 4 --micro-batch 4 --max-completion 64 --no-wandb

  # pod (vLLM): stage-4-like settings from the stage-4 adapter, CISPO with 2 updates per batch
  uv run python scripts/rl_train.py --algo cispo --updates-per-batch 2 --stage 5 --max-completion 512 \
      --max-steps 300 --seed 5 --hops 2 3 4 --mix 0.6 0.2 0.2 --save-steps 25 \
      --resume-adapter data/ckpt/grpo-step-r1-stage4 --onpolicy-every 25 --onpolicy-games 32 --onpolicy-share 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wikigame import WikiGameEnv, WikiGraph
from wikigame.env import RewardConfig
from wikigame.grpo_reward import StepReward
from wikigame.llm import resolve_hint
from wikigame.rl.algorithms import ALGORITHMS, get_algorithm
from wikigame.states import StateMix, sample_states
from wikigame.tracking import detect_machine, init_run

R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def vllm_available() -> bool:
    try:
        import torch
        import vllm  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=R1)
    ap.add_argument("--graph", default="data/wikispeedia.json")
    ap.add_argument("--graph-name", default="wikispeedia")
    ap.add_argument("--stage", default="5", help="label for the run name")
    ap.add_argument("--resume-adapter", default=None, help="start from this LoRA adapter dir (weights only)")
    ap.add_argument("--resume-optimizer", action="store_true",
                    help="also restore optimizer/scheduler state (needs a checkpoint-N dir as --resume-adapter)")
    ap.add_argument("--out", default=None, help="checkpoint dir (default data/ckpt/<experiment>)")
    # algorithm
    ap.add_argument("--algo", default="grpo", choices=sorted(ALGORITHMS))
    ap.add_argument("--eps-low", type=float, default=None, help="override the preset's lower clip")
    ap.add_argument("--eps-high", type=float, default=None, help="override the preset's upper clip")
    ap.add_argument("--cispo-max", type=float, default=None, help="override CISPO's IS-weight cap")
    ap.add_argument("--agg", default=None, choices=["token_mean", "seq_mean", "seq_sum_const"])
    ap.add_argument("--drop-zero-std", default=None, choices=["on", "off"], help="override the preset's zero-std filter")
    ap.add_argument("--updates-per-batch", type=int, default=1,
                    help="optimizer updates per generated batch (1 = on-policy, ratio exactly 1)")
    ap.add_argument("--tis-cap", type=float, default=0.0,
                    help="truncated importance-sampling cap for the vLLM/trainer mismatch (0 = off; TRL uses 3)")
    # where groups come from (v2)
    ap.add_argument("--source", default="step", choices=["step", "playout"],
                    help="step = score each answer by its one move (StepReward, stages 1-4); playout = v2 branch and "
                         "play out: each answer's game is played to the end by the model and scored by its outcome")
    ap.add_argument("--move-cost", type=float, default=0.05, help="playout: a win scores 1 - this x moves used")
    ap.add_argument("--loss-credit", type=float, default=0.3,
                    help="playout: a loss scores this x the share of the distance to the target it closed")
    ap.add_argument("--invalid-end-penalty", type=float, default=0.3,
                    help="playout: subtracted from a loss that ended on too many invalid answers")
    ap.add_argument("--playout-lenient", action="store_true",
                    help="playout: accept bare/```json titles in the play-out moves and the branching move")
    # rollout / optimisation
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"],
                    help="generation: vllm (Linux+CUDA), hf (model.generate), auto = vllm when available")
    ap.add_argument("--vllm-mem", type=float, default=0.35, help="vLLM's share of the GPU (weights + KV cache)")
    ap.add_argument("--vllm-max-len", type=int, default=4096)
    ap.add_argument("--vllm-sleep", action="store_true", help="put vLLM to sleep during each update")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8, help="sequences per forward/backward")
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-schedule", default="linear", choices=["linear", "constant", "cosine"])
    ap.add_argument("--warmup", type=int, default=0, help="warm-up optimizer updates")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    # states / env (as grpo_train.py)
    ap.add_argument("--hops", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--mix", type=float, nargs=3, default=[0.6, 0.2, 0.2], metavar=("START", "OPTIMAL", "NOISY"))
    ap.add_argument("--feedback", action="store_true")
    ap.add_argument("--p-post-invalid", type=float, default=0.0)
    ap.add_argument("--flat-invalid", action="store_true")
    ap.add_argument("--budget-aware", action="store_true")
    ap.add_argument("--sideways-bonus", type=float, default=0.0)
    ap.add_argument("--hint", default=None)
    ap.add_argument("--onpolicy-every", type=int, default=0)
    ap.add_argument("--onpolicy-games", type=int, default=32)
    ap.add_argument("--onpolicy-share", type=float, default=0.5)
    ap.add_argument("--onpolicy-batch", type=int, default=32)
    ap.add_argument("--max-steps-game", type=int, default=10)
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    import random

    import numpy as np
    import torch

    from wikigame.onpolicy import StatePool
    from wikigame.rl.generation import HFGenerator, VLLMGenerator
    from wikigame.rl.learner import Learner
    from wikigame.rl.sources import OnPolicyRounds, PlayoutReturn, PlayoutSource, StepSource
    from wikigame.rl.trainer import RLTrainer, TrainConfig

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    backend = ("vllm" if vllm_available() else "hf") if a.backend == "auto" else a.backend
    alg = get_algorithm(a.algo).with_overrides(
        eps_low=a.eps_low, eps_high=a.eps_high, cispo_max=a.cispo_max, agg=a.agg,
        drop_zero_std=None if a.drop_zero_std is None else a.drop_zero_std == "on")
    machine = detect_machine()
    short = a.model.split("/")[-1].lower().replace("deepseek-r1-distill-qwen-", "r1-").replace("-instruct", "")
    experiment = f"rl-{a.algo}{'-playout' if a.source == 'playout' else ''}-{short}-stage{a.stage}"
    out = Path(a.out or f"data/ckpt/{experiment}")
    hint = resolve_hint(a.hint)

    graph = WikiGraph.load(a.graph)
    mix = StateMix(start=a.mix[0], optimal=a.mix[1], noisy=a.mix[2], hops=tuple(a.hops), p_post_invalid=a.p_post_invalid)
    state_env = WikiGameEnv(graph, max_steps=a.max_steps_game, feedback=a.feedback, seed=a.seed)
    rows = sample_states(graph, int(a.max_steps * a.prompts_per_step * 1.2) + 8, a.model, mix, seed=a.seed,
                         env=state_env, hint=hint)
    print(f"{len(rows)} training states: " + str({k: sum(r['kind'] == k for r in rows) for k in ('start', 'optimal', 'noisy')}))
    pool = StatePool(rows, a.onpolicy_share, seed=a.seed)
    reward_cfg = RewardConfig(shaping=1.0, budget_aware=a.budget_aware, sideways_bonus=a.sideways_bonus,
                              invalid_by_cause={} if a.flat_invalid else RewardConfig().invalid_by_cause)
    reward = StepReward(graph, reward_cfg, max_completion_length=a.max_completion, max_steps=a.max_steps_game,
                        feedback=a.feedback)

    gen = None
    if backend == "vllm":  # start vLLM first: it sizes its KV cache from the memory it finds free
        gen = VLLMGenerator(a.model, gpu_memory_utilization=a.vllm_mem, max_model_len=a.vllm_max_len,
                            max_lora_rank=a.lora_r, seed=a.seed, enable_sleep_mode=a.vllm_sleep,
                            adapter=a.resume_adapter)
        print(f"vLLM ready in {gen.startup_seconds:.0f}s", flush=True)
    learner = Learner(a.model, adapter=a.resume_adapter, lora_r=a.lora_r, lr=a.lr,
                      total_updates=a.max_steps * a.updates_per_batch, lr_schedule=a.lr_schedule,
                      warmup_updates=a.warmup, temperature=a.temperature, max_grad_norm=a.max_grad_norm,
                      grad_checkpointing=not a.no_grad_checkpointing)
    if a.resume_optimizer:
        learner.load_optimizer(a.resume_adapter)
    if gen is None:
        gen = HFGenerator(learner.model, learner.tok)

    if a.source == "playout":
        returns = PlayoutReturn(move_cost=a.move_cost, loss_credit=a.loss_credit,
                                invalid_end_penalty=a.invalid_end_penalty)
        source = PlayoutSource(pool, graph, learner.tok, gen, num_generations=a.num_generations,
                               max_tokens=a.max_completion, temperature=a.temperature, model_id=a.model, hint=hint,
                               max_steps_game=a.max_steps_game, feedback=a.feedback, lenient=a.playout_lenient,
                               returns=returns, seed=a.seed)
    else:
        source = StepSource(pool, learner.tok, gen, reward, num_generations=a.num_generations,
                            max_tokens=a.max_completion, temperature=a.temperature)
    rounds = None
    if a.onpolicy_every:
        rounds = OnPolicyRounds(pool, graph, gen, learner.tok, every=a.onpolicy_every, n_games=a.onpolicy_games,
                                hops=tuple(a.hops), model_id=a.model, hint=hint, max_steps_game=a.max_steps_game,
                                max_tokens=a.max_completion, temperature=a.temperature, batch_size=a.onpolicy_batch,
                                feedback=a.feedback, seed=a.seed,
                                log_path=Path("data/runs") / f"{experiment}-onpolicy-rounds.jsonl")
    cfg = TrainConfig(out=out, max_steps=a.max_steps, prompts_per_step=a.prompts_per_step,
                      num_generations=a.num_generations, micro_batch=a.micro_batch, max_completion=a.max_completion,
                      updates_per_batch=a.updates_per_batch, tis_cap=a.tis_cap, save_steps=a.save_steps,
                      sleep_sampler=a.vllm_sleep)
    run_cfg = dict(vars(a), experiment=experiment, machine=machine, backend=backend, algorithm=vars(alg),
                   reward=vars(reward_cfg), n_train_states=len(rows), trainer="wikigame.rl",
                   completions_per_step=a.prompts_per_step * a.num_generations)
    log_fn = None
    if not a.no_wandb:
        run = init_run(experiment, "train", run_cfg, graph, a.graph_name,
                       tags=["rl", a.algo, f"backend:{backend}", f"source:{a.source}", f"stage:{a.stage}"], group="rl-loop")
        log_fn = lambda m, step: run.log(m, step=step)  # noqa: E731
    trainer = RLTrainer(learner, gen, source, alg, cfg, rounds=rounds, log_fn=log_fn)
    trainer.run()
    (out / "wikigame_config.json").write_text(json.dumps(run_cfg, indent=2, default=str))
    print("adapter saved to", out)
    if hasattr(gen, "stats"):
        print("generation:", gen.stats.summary())
    if log_fn:
        run.finish()


if __name__ == "__main__":
    main()
