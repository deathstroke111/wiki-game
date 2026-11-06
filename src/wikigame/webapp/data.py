"""Data layer for the webapp. W&B is the source of truth; a fixture source backs tests.

Set WIKIGAME_APP_SOURCE=fixture to run the app against synthetic data (used by AppTest).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd

from ..graph import WikiGraph

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "cache"
GRAPH_DIR = ROOT / "data"


@dataclass
class RunInfo:
    id: str
    name: str
    machine: str
    group: str
    job_type: str
    state: str
    tags: list[str]
    created_at: str
    config: dict
    summary: dict[str, float]  # numeric eval/* (and train/*) summary values only

    @property
    def graph(self) -> str | None:
        return self.config.get("graph")


@dataclass
class EpisodeSet:
    """One logged batch of episodes (an `episodes` artifact)."""
    key: str  # artifact name, unique per run
    step: int
    split: str
    n_episodes: int
    digest: str = ""


class DataSource(Protocol):
    def list_runs(self) -> list[RunInfo]: ...
    def run_history(self, run_id: str) -> pd.DataFrame: ...
    def episode_sets(self, run_id: str) -> list[EpisodeSet]: ...
    def load_episodes(self, run_id: str, key: str) -> list[dict]: ...
    def load_graph(self, name: str | None) -> WikiGraph | None: ...


def load_local_graph(name: str | None) -> WikiGraph | None:
    """data/<name>.json if present (the app degrades to text-only paths otherwise)."""
    if not name or "/" in name or ".." in name:
        return None
    p = GRAPH_DIR / f"{name}.json"
    return WikiGraph.load(p) if p.exists() else None


def _machine_of(tags: list[str], config: dict) -> str:
    for t in tags:
        if t.startswith("machine:"):
            return t.split(":", 1)[1]
    return config.get("machine", "unknown")


class WandbSource:
    def __init__(self, entity: str | None = None, project: str | None = None):
        import wandb

        from ..tracking import ENTITY, PROJECT

        self.path = f"{entity or os.environ.get('WANDB_ENTITY', ENTITY)}/{project or os.environ.get('WANDB_PROJECT', PROJECT)}"
        self.api = wandb.Api(timeout=60)

    def list_runs(self) -> list[RunInfo]:
        out = []
        for r in self.api.runs(self.path, order="-created_at"):
            summ = {k: v for k, v in dict(r.summary).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and not k.startswith("_")}
            cfg = {k: v for k, v in r.config.items() if not k.startswith("_")}
            tags = list(r.tags)
            out.append(RunInfo(r.id, r.name, _machine_of(tags, cfg), r.group or "", r.job_type or "",
                               r.state, tags, str(r.created_at), cfg, summ))
        return out

    def run_history(self, run_id: str) -> pd.DataFrame:
        run = self.api.run(f"{self.path}/{run_id}")
        rows = list(run.scan_history())
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # drop table/media columns (dicts) - keep numeric metrics and the step
        keep = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        return df[keep]

    def _artifacts(self, run_id: str):
        run = self.api.run(f"{self.path}/{run_id}")
        return [a for a in run.logged_artifacts() if a.type == "episodes"]

    def episode_sets(self, run_id: str) -> list[EpisodeSet]:
        sets = [EpisodeSet(a.name, int(a.metadata.get("step", 0)), a.metadata.get("split", ""),
                           int(a.metadata.get("n_episodes", 0)), a.digest) for a in self._artifacts(run_id)]
        return sorted(sets, key=lambda s: (s.split, s.step))

    def load_episodes(self, run_id: str, key: str) -> list[dict]:
        art = next(a for a in self._artifacts(run_id) if a.name == key)
        cached = CACHE_DIR / art.digest / "episodes.jsonl"  # content-addressed: never stale
        if not cached.exists():
            art.download(root=str(CACHE_DIR / art.digest))
        return [json.loads(line) for line in cached.read_text().splitlines() if line]

    def load_graph(self, name: str | None) -> WikiGraph | None:
        return load_local_graph(name)


@dataclass
class FixtureSource:
    """In-memory source for tests: run infos, per-run history, per-(run,key) episodes, one graph."""
    runs: list[RunInfo] = field(default_factory=list)
    histories: dict[str, pd.DataFrame] = field(default_factory=dict)
    episodes: dict[tuple[str, str], list[dict]] = field(default_factory=dict)
    sets: dict[str, list[EpisodeSet]] = field(default_factory=dict)
    graph: WikiGraph | None = None

    def list_runs(self):
        return self.runs

    def run_history(self, run_id):
        return self.histories.get(run_id, pd.DataFrame())

    def episode_sets(self, run_id):
        return self.sets.get(run_id, [])

    def load_episodes(self, run_id, key):
        return self.episodes[(run_id, key)]

    def load_graph(self, name):
        return self.graph


def make_fixture_source() -> FixtureSource:
    """Synthetic data: tiny graph, random + oracle eval episodes, and a fake 'training' history."""
    from ..env import WikiGameEnv
    from ..eval import evaluate, make_pairs
    from ..rollout import oracle_policy, random_policy

    nodes = {t: f"About {t}" for t in "ABCDEFGH"}
    edges = {"A": ["B", "C"], "B": ["D", "A"], "C": ["E", "F"], "D": ["G"], "E": ["G", "H"],
             "F": ["H"], "G": ["H"], "H": ["A"]}
    g = WikiGraph.from_edges(nodes, edges)
    pairs = make_pairs(g, hops=(1, 2, 3), n_per_hop=10)
    src = FixtureSource(graph=g)
    for i, (name, pol, machine) in enumerate([("random", random_policy(0), "local"),
                                              ("oracle", oracle_policy(), "cuda")]):
        env = WikiGameEnv(g, max_steps=8, max_links_shown=500, seed=0)
        metrics, eps = evaluate(env, pol, pairs)
        rid = f"run{i}"
        src.runs.append(RunInfo(rid, f"baseline-{name}-{machine}", machine, f"baseline-{name}", "baseline",
                                "finished", [f"machine:{machine}", "graph:fixture"], f"2026-11-0{i + 1}",
                                {"graph": "fixture", "reward": {"step_cost": 0.02}},
                                {f"eval/{k}": v for k, v in metrics.items()}))
        key = f"episodes-{rid}-eval-step0:v0"
        src.sets[rid] = [EpisodeSet(key, 0, "eval", len(eps), "d" + rid)]
        src.episodes[(rid, key)] = [e.to_dict() for e in eps]
        src.histories[rid] = pd.DataFrame([{"_step": s, "train/reward_mean": 0.1 * s * (i + 1),
                                            "train/success_rate": min(1.0, 0.05 * s * (i + 1))} for s in range(10)])
    return src


def get_source() -> DataSource:
    if os.environ.get("WIKIGAME_APP_SOURCE") == "fixture":
        return make_fixture_source()
    return WandbSource()
