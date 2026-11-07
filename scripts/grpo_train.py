"""Per-step GRPO with TRL + LoRA on the Wikipedia game.

One training example = one game state. The model samples G answers (think + <answer>{"title": ...}</answer>),
each answer is scored by taking that move in the env (progress-shaped reward, invalid = -2), and TRL turns the
group-relative reward into the policy update. Eval is separate: checkpoints are LoRA adapters, evaluated on the
held-out pairs with  scripts/zero_shot.py --adapter <dir>  (same pairs/metrics as every baseline).

  # plumbing check on the Mac (tiny model, tiny everything; the numbers mean nothing)
  HF_HOME=data/hf uv run python scripts/grpo_train.py --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 3 \
      --num-generations 4 --prompts-per-step 1 --micro-batch 4 --max-completion 128 --no-wandb

  # stage 1 on the RTX (R1-1.5B, 1024-token cap), then stage 2 resumes the adapter with a 512 cap
  HF_HOME=data/hf uv run python scripts/grpo_train.py --stage 1 --max-completion 1024 --max-steps 300
  HF_HOME=data/hf uv run python scripts/grpo_train.py --stage 2 --max-completion 512 --max-steps 300 \
      --resume-adapter data/ckpt/grpo-step-r1-stage1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wikigame import WikiGraph
from wikigame.env import RewardConfig
from wikigame.grpo_reward import StepReward
from wikigame.states import StateMix, sample_states
from wikigame.tracking import detect_machine, init_run

R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=R1)
    ap.add_argument("--graph", default="data/wikispeedia.json")
    ap.add_argument("--graph-name", default="wikispeedia")
    ap.add_argument("--stage", default="1", help="label for the length-curriculum stage (part of the run name)")
    ap.add_argument("--resume-adapter", default=None, help="continue training from this LoRA adapter dir")
    ap.add_argument("--out", default=None, help="checkpoint dir (default data/ckpt/<experiment>)")
    # rollout / optimisation
    ap.add_argument("--max-steps", type=int, default=300, help="optimizer steps")
    ap.add_argument("--num-generations", type=int, default=8, help="G: answers sampled per prompt")
    ap.add_argument("--prompts-per-step", type=int, default=4, help="distinct prompts (groups) per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=4, help="sequences per forward/backward (lower if OOM)")
    ap.add_argument("--max-completion", type=int, default=1024, help="completion token cap (cut-off = no answer = invalid)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0, help="KL weight to the reference policy (0 = no ref model)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--dtype", default="auto", help="auto = bfloat16 on CUDA, float32 elsewhere")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--save-total-limit", type=int, default=0,
                    help="keep only the newest N checkpoints (0 = keep all, so every one can be evaluated later)")
    ap.add_argument("--seed", type=int, default=0)
    # states / env
    ap.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--mix", type=float, nargs=3, default=[0.4, 0.3, 0.3], metavar=("START", "OPTIMAL", "NOISY"))
    ap.add_argument("--feedback", action="store_true", help="show the invalid-answer feedback line (env flag)")
    ap.add_argument("--p-post-invalid", type=float, default=0.0, help="share of states right after an invalid answer")
    ap.add_argument("--flat-invalid", action="store_true",
                    help="ablation: every invalid answer scores the same -2 instead of the ranked tiers")
    ap.add_argument("--max-steps-game", type=int, default=10)
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    machine = detect_machine()
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (a.dtype == "bfloat16" or (a.dtype == "auto" and use_cuda)) else torch.float32
    short = a.model.split("/")[-1].lower().replace("deepseek-r1-distill-qwen-", "r1-").replace("-instruct", "")
    experiment = f"grpo-step-{short}-stage{a.stage}"
    out = Path(a.out or f"data/ckpt/{experiment}")
    per_step = a.prompts_per_step * a.num_generations
    if per_step % a.micro_batch:
        raise SystemExit(f"prompts-per-step x num-generations ({per_step}) must be divisible by --micro-batch ({a.micro_batch})")

    graph = WikiGraph.load(a.graph)
    mix = StateMix(start=a.mix[0], optimal=a.mix[1], noisy=a.mix[2], hops=tuple(a.hops), p_post_invalid=a.p_post_invalid)
    from wikigame import WikiGameEnv
    state_env = WikiGameEnv(graph, max_steps=a.max_steps_game, feedback=a.feedback, seed=a.seed)
    rows = sample_states(graph, int(a.max_steps * a.prompts_per_step * 1.2) + 8, a.model, mix, seed=a.seed, env=state_env)
    dataset = Dataset.from_list(rows)
    print(f"{len(rows)} training states: " + str({k: sum(r['kind'] == k for r in rows) for k in ('start', 'optimal', 'noisy')}))

    reward_cfg = RewardConfig(shaping=1.0, invalid_by_cause={} if a.flat_invalid else RewardConfig().invalid_by_cause)
    reward_fn = StepReward(graph, reward_cfg, max_completion_length=a.max_completion, max_steps=a.max_steps_game,
                           feedback=a.feedback)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)
    peft_config = None
    if a.resume_adapter:
        model = PeftModel.from_pretrained(model, a.resume_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, target_modules="all-linear",
                                 task_type="CAUSAL_LM")

    cfg = dict(vars(a), experiment=experiment, machine=machine, dtype=str(dtype), reward=vars(reward_cfg),
               n_train_states=len(rows), completions_per_step=per_step, trainer="trl.GRPOTrainer", loss="dapo")
    if not a.no_wandb:
        init_run(experiment, "train", cfg, graph, a.graph_name, tags=["grpo", "per-step", f"stage:{a.stage}"],
                 group="grpo-per-step")

    class Telemetry(TrainerCallback):
        """Peak GPU memory in the logs, so we can size --micro-batch / --max-completion on the RTX."""

        def on_log(self, args, state, control, logs=None, **kw):
            if logs is not None and use_cuda:
                logs["sys/cuda_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30

    args = GRPOConfig(
        output_dir=str(out), max_steps=a.max_steps, learning_rate=a.lr, beta=a.beta,
        num_generations=a.num_generations, max_completion_length=a.max_completion, temperature=a.temperature,
        per_device_train_batch_size=a.micro_batch, gradient_accumulation_steps=per_step // a.micro_batch,
        gradient_checkpointing=True, bf16=use_cuda and dtype == torch.bfloat16, logging_steps=1,
        save_steps=a.save_steps, save_total_limit=a.save_total_limit or None, seed=a.seed, remove_unused_columns=False,
        report_to="none" if a.no_wandb else "wandb", log_completions=not a.no_wandb,
        model_init_kwargs=None,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=args, train_dataset=dataset,
                          processing_class=tok, peft_config=peft_config, callbacks=[Telemetry()])
    trainer.train()
    trainer.save_model(str(out))
    (out / "wikigame_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print("adapter saved to", out)
    if not a.no_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
