"""Offline Wikipedia link graph + shortest-path utilities.

Nodes are articles (int ids), edges are outgoing links. Each node stores a
title and a short summary. Matches the graph model in "Can Language Models Play
the Wikipedia Game?" (directed graph, title + summary per node).
"""
from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WikiGraph:
    titles: list[str]
    summaries: list[str]
    adj: list[list[int]]  # adj[i] = outgoing link ids of node i (deduped, no self loops)
    _title_to_id: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        assert len(self.titles) == len(self.summaries) == len(self.adj)
        self._title_to_id = {t: i for i, t in enumerate(self.titles)}

    def __len__(self) -> int:
        return len(self.titles)

    def id_of(self, title: str) -> int | None:
        return self._title_to_id.get(title)

    def bfs_dist(self, src: int, max_depth: int | None = None) -> dict[int, int]:
        """Hop distance from src to every reachable node (optionally depth-capped)."""
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            if max_depth is not None and dist[u] >= max_depth:
                continue
            for v in self.adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def shortest_dist(self, src: int, dst: int) -> int | None:
        return self.bfs_dist(src).get(dst)

    def shortest_path(self, src: int, dst: int) -> list[int] | None:
        """One shortest path src -> dst as node ids (BFS with parent pointers), or None if unreachable."""
        parent = {src: -1}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v in self.adj[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        if dst not in parent:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(parent[path[-1]])
        return path[::-1]

    def dist_to(self, dst: int) -> dict[int, int]:
        """Hop distance from every node TO dst (BFS on the reversed graph). Used by the oracle policy."""
        if not hasattr(self, "_radj"):
            radj: list[list[int]] = [[] for _ in self.adj]
            for u, outs in enumerate(self.adj):
                for v in outs:
                    radj[v].append(u)
            self._radj = radj
        dist = {dst: 0}
        q = deque([dst])
        while q:
            u = q.popleft()
            for v in self._radj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def sample_pair(self, n_hops: int, rng: random.Random, max_tries: int = 1000) -> tuple[int, int]:
        """Sample (source, target) whose shortest path is exactly n_hops (the curriculum knob)."""
        candidates = [i for i, out in enumerate(self.adj) if out]
        for _ in range(max_tries):
            src = rng.choice(candidates)
            at_k = [v for v, d in self.bfs_dist(src, max_depth=n_hops).items() if d == n_hops]
            if at_k:
                return src, rng.choice(at_k)
        raise RuntimeError(f"no pair at distance {n_hops} found in {max_tries} tries")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"titles": self.titles, "summaries": self.summaries, "adj": self.adj}))

    @classmethod
    def load(cls, path: str | Path) -> "WikiGraph":
        d = json.loads(Path(path).read_text())
        return cls(d["titles"], d["summaries"], d["adj"])

    @classmethod
    def from_edges(cls, nodes: dict[str, str], edges: dict[str, list[str]]) -> "WikiGraph":
        """Build from {title: summary} and {title: [linked titles]}; drops dangling/self/duplicate links."""
        titles = list(nodes)
        idx = {t: i for i, t in enumerate(titles)}
        adj = []
        for t in titles:
            seen: dict[int, None] = {}
            for o in edges.get(t, []):
                j = idx.get(o)
                if j is not None and j != idx[t]:
                    seen[j] = None
            adj.append(list(seen))
        return cls(titles, [nodes[t] for t in titles], adj)
