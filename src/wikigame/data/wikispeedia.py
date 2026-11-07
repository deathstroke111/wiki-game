"""Build a WikiGraph from the Wikispeedia (SNAP) release.

Inputs (extracted by the caller, never executed):
  graph_dir/articles.tsv, graph_dir/links.tsv     (URL-encoded titles, '#' comment lines)
  text_dir/<url-encoded title>.txt                (plaintext articles)
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

from ..graph import WikiGraph

_WS = re.compile(r"\s+")
_SENT_END = re.compile(r"(?<=[.!?])\s")


def _read_tsv(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            rows.append(line.split("\t"))
    return rows


def _title(raw: str) -> str:
    return unquote(raw).replace("_", " ")


def _looks_like_prose(p: str) -> bool:
    """Reject infoboxes/captions: need sentence punctuation and mostly lowercase words."""
    if "Image:" in p or "Enlarge" in p or not re.search(r"[a-z]{3,}[,;]? [a-z]{2,}", p):
        return False
    words = p.split()
    lower = sum(w[0].islower() for w in words if w[0].isalpha())
    return lower / max(1, len(words)) >= 0.45 and bool(_SENT_END.search(p) or p.endswith("."))


def extract_summary(text: str, max_chars: int = 500) -> str:
    """First real prose paragraph: skip the copyright/title/boilerplate header, captions, and stubs."""
    for para in re.split(r"\n\s*\n", text):
        p = _WS.sub(" ", para).strip()
        if len(p) < 100:  # '#copyright', title, 'Enlarge', short captions
            continue
        if p.startswith("2007 Schools Wikipedia Selection") or not _looks_like_prose(p):
            continue
        if len(p) <= max_chars:
            return p
        cut = p[:max_chars]
        ends = [m.start() for m in _SENT_END.finditer(cut)]
        return cut[: ends[-1]] if ends and ends[-1] > max_chars // 3 else cut.rstrip() + "…"
    return ""


def build_wikispeedia(graph_dir: str | Path, text_dir: str | Path) -> WikiGraph:
    graph_dir, text_dir = Path(graph_dir), Path(text_dir)
    raw_titles = [r[0] for r in _read_tsv(graph_dir / "articles.tsv")]

    nodes: dict[str, str] = {}
    for raw in raw_titles:
        f = text_dir / f"{raw}.txt"
        summary = extract_summary(f.read_text(encoding="utf-8", errors="replace")) if f.exists() else ""
        nodes[_title(raw)] = summary

    edges: dict[str, list[str]] = {}
    for src, dst in (r[:2] for r in _read_tsv(graph_dir / "links.tsv")):
        edges.setdefault(_title(src), []).append(_title(dst))
    return WikiGraph.from_edges(nodes, edges)
