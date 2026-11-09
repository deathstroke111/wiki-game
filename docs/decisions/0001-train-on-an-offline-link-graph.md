---
status: "accepted"
date: 2026-11-06
decision-makers: "project owner (user), dev agent (Claude)"
consulted: "arXiv 2610.13948 (benchmark, no RL); `aigym` (proprietary, only used as inspiration)"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Train and evaluate on an offline link graph, not live Wikipedia

## Context and Problem Statement

The goal is to reproduce an RL pipeline for the Wikipedia game (navigate from a start article to a target by clicking links), inspired by arXiv 2610.13948 (a benchmark of LLMs playing the game, no RL) and the `aigym` project. `aigym` is proprietary, so all code here is our own. We had to decide what world the agent plays in: the live website, or a frozen snapshot.

The user's stated preference (2026-11-06) was to have "the model playing over a dump rather than the entire online learning over entire wikipedia".

## Decision Drivers

* RL needs many cheap, repeatable rollouts; each GRPO step replays dozens of moves.
* Exact shortest-path distance (BFS) is needed for the oracle baseline, the held-out hop curriculum and the shaped reward ([ADR-0010](0010-per-step-progress-shaped-reward.md)).
* Reproducibility: results must be comparable across the Mac and the rented GPU box.
* No dependence on network access during training (the rented box may be offline; `WANDB_MODE=offline` is supported for the same reason).

## Considered Options

* Play against live (online) Wikipedia.
* Play over an offline snapshot ("dump") of the link graph.

## Decision Outcome

Chosen option: "offline snapshot", because it makes episodes deterministic and fast, gives exact BFS distances for every state, and removes rate limits and network flakiness. Which snapshot is a separate decision: [ADR-0002](0002-wikispeedia-first-dataset.md).

### Consequences

* Good, because every episode can be replayed exactly (`WikiGameEnv.set_state` rebuilds any mid-game state, used by the GRPO reward in [ADR-0012](0012-trl-grpo-with-lora-per-step.md)).
* Good, because oracle (100%) and random baselines are computable exactly.
* Bad, because the graph is small and static (see ADR-0002), so results do not directly say how the agent would do on the real, current Wikipedia.
* Neutral, because paper numbers are not directly comparable (different graph); verifying the paper's table is still an open item in CLAUDE.md.

## Implementation Plan

* **Affected paths**: `src/wikigame/graph.py` (`WikiGraph`: titles, summaries, links, BFS, hop-exact pair sampling), `src/wikigame/env.py`.
* **Patterns to follow**: all graph access goes through `WikiGraph` (`id_of`, `adj`, `shortest_dist`, `dist_to`).
* **Patterns to avoid**: no live network calls inside env/rollout/eval code.

### Verification

- [ ] `uv run pytest tests/test_env.py` passes (BFS and exact-hop pair sampling are covered by `test_bfs_and_pairs`).
- [ ] `grep -rn "requests\|urllib" src/wikigame/env.py src/wikigame/rollout.py src/wikigame/eval.py` finds nothing.

## More Information

* Provenance: decision taken in the first planning turns on 2026-11-06; recorded retrospectively on 2026-11-10 from the session transcript and CLAUDE.md.
* Revisit if we want to claim results on real Wikipedia; the open item "Simple English Wikipedia loader" in CLAUDE.md is the next step in that direction.
