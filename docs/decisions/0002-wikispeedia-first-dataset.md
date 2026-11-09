---
status: "accepted"
date: 2026-11-06
decision-makers: "project owner (user), dev agent (Claude)"
consulted: "-"
informed: "CUDA-box agent (via CLAUDE.md)"
---

# Use the SNAP Wikispeedia graph first; defer Simple English Wikipedia

## Context and Problem Statement

[ADR-0001](0001-train-on-an-offline-link-graph.md) fixed "offline graph"; we still needed a concrete dataset. The user asked "which dataset is ideal for this?" (2026-11-06). Two tiers were identified: a small ready-made graph (Wikispeedia) and a bigger one built from a Simple English Wikipedia dump.

## Decision Drivers

* Get an end-to-end pipeline (env, baselines, app, tracking) running before investing in data engineering.
* Need page text (a summary per page) as well as links, since the model sees the target's summary.
* A human baseline would be nice to have (Wikispeedia ships human play data).
* Keep the download small; downloaded data is untrusted and must be handled with `python -I` and safe extraction.

## Considered Options

* Wikispeedia (SNAP): 4,604 articles, 119,772 links, ready-made plaintext articles.
* Simple English Wikipedia dump (SQL + XML, a few hundred MB to parse).
* Full English Wikipedia (not seriously pursued).

## Decision Outcome

Chosen option: "Wikispeedia first", because it needs no large download, matches the published graph statistics (we verified 4,604 nodes / 119,772 edges, median out-degree 19, 17 dead-end pages) and unblocked everything else. The Simple English loader is explicitly deferred (CLAUDE.md "Open items").

### Consequences

* Good, because the whole pipeline was built and tested in a day on the Mac.
* Good, because sampled pairs hit the exact requested hop distance for 1-4 hops.
* Bad, because the graph is small, so models may overfit its structure; hop 4 pairs are rare, and eval pools are limited (50 pairs per hop, see [ADR-0005](0005-heldout-evaluation-protocol.md)).
* Bad, because some page summaries start mid-article (e.g. "Brain"); the heuristic takes the first prose paragraph and caps it at 500 characters.
* Follow-up: Simple English loader, TF-IDF/embedding baselines.

## Implementation Plan

* **Affected paths**: `src/wikigame/data/` (`wikispeedia.py`, `__main__.py`), `tests/test_wikispeedia.py`, `data/raw/wikispeedia_graph` and `data/raw/wikispeedia_text` (gitignored inputs), `data/wikispeedia.json` (gitignored output).
* **Command**: `uv run python -I -m wikigame.data wikispeedia [--raw data/raw] [--out data/wikispeedia.json]`.
* **Patterns to follow**: read raw downloads only with `python -I` and extract archives with `tarfile`'s `data` filter.
* **Gotcha (learned the hard way)**: `.gitignore` must say `/data/`, not `data/`; the bare form also ignores `src/wikigame/data/`, so the loader never reached GitHub until 2026-11-08 (commit 8fb640a).

### Verification

- [ ] `git ls-files src/wikigame/data` lists three files.
- [ ] `grep -n "^/data/" .gitignore` matches.
- [ ] `uv run pytest tests/test_wikispeedia.py` passes (the real-graph smoke test skips if `data/wikispeedia.json` is missing).

## More Information

Provenance: assistant analysis of 2026-11-06 08:38 (Tier 2 built and tested); recorded retrospectively on 2026-11-10.
