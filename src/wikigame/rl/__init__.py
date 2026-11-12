"""Our own RL training loop (v3): vLLM or HF generation, pluggable policy-gradient algorithms, a LoRA learner.

  algorithms  advantage + objective + aggregation presets (reinforce, rloo, grpo, dapo, trl-dapo, cispo, gspo)
  generation  Generator interface: VLLMGenerator (pod), HFGenerator (Mac / fallback), ScriptedGenerator (tests)
  learner     LoRA model, per-token log-probs, optimizer, checkpoints
  sources     game states -> groups of scored answers (StepSource), on-policy rounds
  trainer     the loop (scripts/rl_train.py drives it)
"""
from .algorithms import ALGORITHMS, Algorithm, get_algorithm

__all__ = ["ALGORITHMS", "Algorithm", "get_algorithm"]
