"""python -I -m wikigame.data wikispeedia [--raw data/raw] [--out data/wikispeedia.json]"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from .wikispeedia import build_wikispeedia


def stats(g) -> str:
    deg = [len(a) for a in g.adj]
    return (
        f"nodes={len(g)} edges={sum(deg)} median_out_degree={statistics.median(deg)} "
        f"max_out_degree={max(deg)} dead_ends={sum(d == 0 for d in deg)} "
        f"empty_summaries={sum(not s for s in g.summaries)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="wikigame.data")
    ap.add_argument("name", choices=["wikispeedia"])
    ap.add_argument("--raw", default="data/raw", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    if a.name == "wikispeedia":
        g = build_wikispeedia(
            a.raw / "wikispeedia_graph" / "wikispeedia_paths-and-graph",
            a.raw / "wikispeedia_text" / "plaintext_articles",
        )
    out = a.out or Path("data") / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    g.save(out)
    print(stats(g))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
