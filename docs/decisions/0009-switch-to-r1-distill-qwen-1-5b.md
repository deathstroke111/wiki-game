---
status: "accepted"
date: 2026-11-07
decision-makers: "project owner (user)"
consulted: "dev agent (Claude) provided speed and format evidence"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Use a ~1.5B model; switch from Qwen2.5-1.5B-Instruct to DeepSeek-R1-Distill-Qwen-1.5B

## Context and Problem Statement

Two linked decisions. First (2026-11-06) the user chose "model size with 1.5B": small enough for LoRA training in 16 GB, big enough to follow a JSON tag format. The assistant's default pick was Qwen2.5-1.5B-Instruct ("my default pick, not a measured choice"). Second (2026-11-07), after the 50-per-hop baselines ([ADR-0008](0008-baselines-at-50-pairs-per-hop.md)) the user wrote that the Qwen run "looks like its worse because the model itself is quite bad at instruction following" and asked to switch to DeepSeek-R1-Distill-Qwen-1.5B.

## Decision Drivers

* Qwen-Instruct produced ~60% invalid actions (format and legality).
* A reasoning model that thinks before answering might follow the action format better and navigate better.
* Cost: R1 needs ~600-800 tokens per answer, ~25 s per call on the Mac (MPS) versus ~1 s for Qwen, and about half the answers were cut off at 1024 tokens.

## Considered Options

* Stay with Qwen2.5-1.5B-Instruct.
* DeepSeek-R1-Distill-Qwen-1.5B (chosen).
* Make R1 cheaper first: a probe with a prompt hint asking it to reason efficiently, a 1024-token cap, on 4 games (the user asked for it, then said "never mind, keep the deepseek r1 model for now").

## Decision Outcome

Chosen option: "R1-Distill-Qwen-1.5B" (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`), with these handling rules in `llm.py`:

* The system prompt plus a short brevity hint (`R1_HINT`) is folded into the user turn (R1 distills work best without a system prompt).
* The chat template ends the prompt with `<think>\n`; the opening tag is restored in stored output so the parser and diagnostics see a complete think block.
* Sampling at T=0.6 (greedy loops); training uses T=0.7.
* **No R1 baseline before training**, by the owner's choice.

### Consequences

* Good, because training could start from a model that does emit reasoning, and GRPO has plenty to optimise.
* Bad, because untrained R1 is *worse* at the game than Qwen: on the CUDA evals untrained R1 scores 13.8% (invalid 73%, unparseable 68%) versus Qwen's 38% zero-shot. It rambles to the token cap without emitting `<answer>`.
* Bad, because evaluation is slow on the Mac (~16-25 s per call), so adapter evals run on the GPU box.
* Neutral, because after GRPO the trained R1 reaches 35.0%: roughly on par with Qwen zero-shot, not clearly better (different n, decoding and model; do not over-read).
* Lesson: "better at instruction following" was a hypothesis, not a measurement; a cheap R1 baseline would have set expectations. See [LESSONS.md](../LESSONS.md).

## Implementation Plan

* **Affected paths**: `src/wikigame/llm.py` (`R1_HINT`, `is_r1`, `merge_system`, `training_messages`, `HFPolicy`), `scripts/zero_shot.py` (`--model`, `--temperature`, `--adapter`), `tests/test_llm.py`.
* **Patterns to avoid**: always pass `--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` for adapter evals; the script's default is Qwen and would apply an R1 LoRA to the wrong base.
* **Patterns to follow**: tokenize with `add_special_tokens=False`; prepend `<think>\n` when the template ends with it.

### Verification

- [ ] `uv run pytest tests/test_llm.py` passes.
- [ ] A 3-pairs-per-hop smoke eval of the stage-2 adapter loads and plays (done 2026-11-09: 33% success, invalid 37%).

## More Information

Provenance: user messages 2026-11-06 11:10 and 2026-11-07 19:56; AskUserQuestion on R1 speed (2026-11-07 20:03). Recorded retrospectively on 2026-11-10.
