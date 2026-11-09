---
status: "accepted"
date: 2026-11-10
decision-makers: "project owner (user); implemented by the dev agent (Claude) in commit e999bb0. The request itself is not in the transcript this record was built from."
consulted: "-"
informed: "agents on any machine (via the repo)"
---

# Stop tracking CLAUDE.md; agent working notes stay local

## Context and Problem Statement

`CLAUDE.md` carried three things in git: the owner's standing instructions (work in the project venv, check the machine, working style), running notes between two agents (Mac dev agent and the CUDA-box agent, with `STATUS_*` lines polled through `git fetch`), and a growing state-of-play log. The two-agent handoff finished on 2026-11-09 ([ADR-0018](0018-wandb-artifact-handoff.md)), and the project moved to a Runpod pod managed from the Mac ([ADR-0021](0021-try-runpod-on-an-experiment-branch.md)). The commit message of `e999bb0` gives the reason: "The two-agent (Mac <-> CUDA box) handoff it carried is finished and the project now runs on a Runpod pod managed from the Mac. The file stays on disk and is ignored; earlier versions remain in history."

## Decision Drivers

* The handoff protocol it existed for is over; tracked churn in a notes file clutters history (12 of the 33 commits before this change touched it).
* Working notes contain local state (pids, paths, run ids) that go stale quickly.
* Tracked notes are visible to anyone who clones the repo.

## Considered Options

* Keep tracking `CLAUDE.md` as is.
* Untrack it and ignore it; keep it as a local working file (chosen).
* Split it: tracked, durable guidance (rules, conventions, decisions) plus a local-only notes file. *Not decided; see consequences.*

## Decision Outcome

Chosen option: "untrack and ignore". `CLAUDE.md` was removed from the index (`git rm --cached`-style, 179 lines) and `.gitignore` gained a `CLAUDE.md` entry with the comment "Agent working notes: local only, not tracked". The file stays on disk. Its history is still readable, e.g. `git show 9189b84:CLAUDE.md`.

### Consequences

* Good, because history no longer fills with handoff and status commits, and local state can live in the file without being published.
* Bad, because a fresh clone, a new worktree or a rented pod does **not** contain the standing instructions or the run log; the rules in [ADR-0019](0019-working-agreements.md) and the lessons in [LESSONS.md](../LESSONS.md) are now the tracked copy of what matters. Anything an agent on a pod needs must be delivered another way (copied in, or moved into tracked docs).
* Bad, because git-polled `STATUS_*` handoffs no longer work; the W&B-artifact mechanism of ADR-0018 still does, but needs another signal (a W&B run tag, an artifact alias, or the user).
* Bad, because several ADRs cite CLAUDE.md as provenance. They remain accurate as history; read them via `git show 9189b84:CLAUDE.md`.
* Follow-up: decide whether to keep durable rules (venv, machine check, ask before deleting or provisioning) in a tracked file such as `docs/` or the README so new clones inherit them.

## Implementation Plan

* **Affected paths**: `.gitignore` (`CLAUDE.md`), `CLAUDE.md` (local only).
* **Patterns to follow**: put decisions in `docs/decisions/`, durable guidance in tracked docs, and ephemeral state in the local `CLAUDE.md`.
* **Patterns to avoid**: do not re-add `CLAUDE.md` to git without a new ADR; do not rely on it for anything a fresh clone must know.

### Verification

- [ ] `git ls-files CLAUDE.md` prints nothing and `git check-ignore -v CLAUDE.md` matches `.gitignore`.
- [ ] `git show 9189b84:CLAUDE.md | head -3` still prints the old contents.

## More Information

Provenance: commit e999bb0 (2026-11-10 00:55 IST), branch `main`. Recorded on 2026-11-10.
