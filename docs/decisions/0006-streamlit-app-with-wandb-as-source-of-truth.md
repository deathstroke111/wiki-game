---
status: "accepted"
date: 2026-11-06
decision-makers: "project owner (user)"
consulted: "dev agent (Claude) recommended both options"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Streamlit analysis app reading Weights & Biases as the single source of truth

## Context and Problem Statement

The user wanted (2026-11-06) to see agent statistics in a web app and to monitor live training metrics on Weights & Biases cloud, for runs from both the Mac (small batches) and the rented RTX box, "under distinct runs".

## Decision Drivers

* One place that shows runs from both machines.
* Episode-level inspection (path vs optimal path, where it went wrong), not only scalar curves.
* Low maintenance: this is a research tool, not a product.

## Considered Options

* Streamlit app (chosen, recommended by the assistant in the question dialog).
* A heavier web stack. *The exact alternatives shown in the dialog were not preserved.*
* Data source: W&B as the source of truth (chosen) versus local files / a shared store the app owns.

## Decision Outcome

Chosen option: "Streamlit app + W&B source of truth".

* Every run logs to team **aiwizards**, project **wiki-game**; run name `<experiment>-<machine>-<sha>`, tags `machine:*`, `graph:*` (`tracking.py`).
* Episodes are logged as a W&B table and as a JSONL artifact so the app can reload the raw model text.
* The app reads W&B through a `DataSource` abstraction (`WIKIGAME_APP_SOURCE=fixture` for tests). Pages: Runs, Episodes, Failures, Diagnostics, Baselines, Play.

### Consequences

* Good, because Mac and CUDA runs appear together and the same code analyses both.
* Good, because `DataSource` lets AppTest run offline with fixtures.
* Bad, because the app needs network access and W&B credentials (`~/.netrc`); an offline box needs `wandb sync`.
* Bad, because W&B runs are hard to clean up safely (see [ADR-0019](0019-working-agreements.md), never delete without asking).
* Lesson: the Mac app lost its dependencies once because of `uv sync` group handling ([ADR-0004](0004-stack-and-two-machine-workflow.md)).

## Implementation Plan

* **Affected paths**: `app/streamlit_app.py`, `app/views/*.py`, `src/wikigame/webapp/` (data layer, `analysis.py`, `viz.py`, `st_helpers.py`), `src/wikigame/tracking.py`.
* **Command**: `uv run streamlit run app/streamlit_app.py`; stop with `pkill -f "streamlit run app/streamlit_app.py"`.
* **Patterns to follow**: new analysis goes into `webapp/analysis.py` (testable) and the pages stay thin.

### Verification

- [ ] `uv run pytest tests/test_app_analysis.py tests/test_app_pages.py` passes with `--group app` installed.
- [ ] All six pages load without exceptions against live W&B data (checked with `AppTest` on 2026-11-09).

## More Information

Provenance: AskUserQuestion answers on 2026-11-06 08:53. Recorded retrospectively on 2026-11-10.
