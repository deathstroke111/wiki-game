---
status: "accepted"
date: 2026-11-06
decision-makers: "project owner (user)"
consulted: "-"
informed: "all agents on the project (via CLAUDE.md)"
---

# Working agreements between the project owner and the agents

## Context and Problem Statement

Agents act on a real repo, a shared W&B project and a rented GPU. The owner's preferences about pace, consultation, destructive actions and attribution needed to be explicit so that new sessions (and the agent on the other machine) follow them.

## Decision Drivers

* Learning project: the owner wants to understand each design choice, not receive a finished black box.
* Irreversible actions (deleting runs, pushing, long GPU jobs that bill) need explicit consent.
* Sessions end (context runs out) and agents are replaced; the rules must live in the repo.

## Considered Options

* Let the agent decide and report afterwards.
* Written agreements in `CLAUDE.md`, enforced by asking before the risky step (chosen).

## Decision Outcome

Chosen option: "written agreements in CLAUDE.md":

* **Go slowly; commit at checkpoints; consult before building or making design choices.** Present options with trade-offs and context; the owner answers, then the agent builds. (The 2026-11-07 request was "ask me these questions with more context and options to make informed decisions".)
* Commit only when asked; push only when asked. Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
* Run and install everything in the project environment only; check which machine the session is on ([ADR-0004](0004-stack-and-two-machine-workflow.md)).
* **Never delete W&B runs or other data without asking.** Approved exceptions are recorded: the CUDA agent deleted the crashed `k8mnksk6` and the 3-step aborted `pm3m0btf` with approval; `ceebhgej` (first stage-1 run, steps 1-32) was kept as the only record of the slowdown; the Mac's early duplicates `henp9oon` and `pumln526` remain undecided.
* Credentials never enter the repo (`~/.netrc`, or `RUNPOD_API_KEY` as an environment variable only).
* Heavy or billing actions (training runs, evals, provisioning pods) need the owner's go-ahead.
* Evals run only after training finishes, never alongside it.

### Consequences

* Good, because new sessions can continue from CLAUDE.md alone and the owner stays in control of design.
* Good, because destructive cleanups were all explicitly approved and recorded.
* Bad, because the workflow is slower and the owner is asked many questions; some choices (such as the 50-per-hop R1 baseline) are consciously deferred.
* Bad, because CLAUDE.md grows long; it was condensed once (commit add7493) and still needs pruning as decisions move into ADRs.

## Implementation Plan

* **Affected paths**: `CLAUDE.md` (sections "Working style", "HANDOFF", "DEV MACHINE state of play"), `.claude/settings.json`.
* **Patterns to follow**: on a decision point use a question dialog with context and options; record the answer here under `docs/decisions/`.

### Verification

- [ ] `git log --format=%B -5` shows the Co-Authored-By trailer on agent commits.
- [ ] Pushes to `origin` happen only after an explicit request (the one recorded push was the CLAUDE.md handoff notes, commit 34a941e, 2026-11-07).

## More Information

Provenance: CLAUDE.md "Working style (the user's preferences)" and repeated user instructions across the sessions. Recorded retrospectively on 2026-11-10.

**Amendment 2026-11-10:** CLAUDE.md is no longer tracked in git ([ADR-0022](0022-stop-tracking-claude-md.md)); it stays on the owner's disk. The agreements above still stand, but a fresh clone, a new worktree or a rented pod will not see them. This ADR is now the tracked copy of them. The old text is at `git show 9189b84:CLAUDE.md`. Open question for the owner: also mirror the short durable rules (project venv only, check the machine, ask before deleting or provisioning) into a tracked file.
