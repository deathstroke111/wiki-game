"""Weights & Biases tracking: run identity, eval metrics, and episode logging.

Both machines log to the same project (entity ``aiwizards``, project ``wiki-game``) as distinct runs,
told apart by the ``machine:*`` tag and the machine name embedded in the run name.
Overrides: WANDB_ENTITY, WANDB_PROJECT, WIKIGAME_MACHINE (local|cuda). Offline: WANDB_MODE=offline.
Credentials come from `wandb login` / WANDB_API_KEY, never from this repo.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import wandb

from .graph import WikiGraph
from .rollout import Episode

ENTITY = "aiwizards"
PROJECT = "wiki-game"


def detect_machine() -> str:
    """'local' (macOS dev box) or 'cuda' (rented NVIDIA box). Mirrors .claude/hooks/detect-machine.sh."""
    forced = os.environ.get("WIKIGAME_MACHINE")
    if forced:
        if forced not in ("local", "cuda"):
            raise ValueError(f"WIKIGAME_MACHINE must be 'local' or 'cuda', got {forced!r}")
        return forced
    if platform.system() in ("Linux", "Windows") and shutil.which("nvidia-smi"):
        try:
            if subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0:
                return "cuda"
        except (subprocess.SubprocessError, OSError):
            pass
    return "local"


def git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10).stdout
        return sha + ("+dirty" if dirty.strip() else "")
    except (subprocess.SubprocessError, OSError):
        return "nogit"


def init_run(
    experiment: str,
    job_type: str,
    config: dict,
    graph: WikiGraph | None = None,
    graph_name: str | None = None,
    tags: list[str] | None = None,
    machine: str | None = None,
    group: str | None = None,
):
    """Start a W&B run. job_type: train | eval | baseline | smoke. Returns the wandb run."""
    machine = machine or detect_machine()
    sha = git_sha()
    cfg = {**config, "machine": machine, "git_sha": sha, "experiment": experiment}
    all_tags = [f"machine:{machine}", *(tags or [])]
    if graph is not None:
        cfg.update(graph_nodes=len(graph), graph_edges=sum(map(len, graph.adj)))
    if graph_name:
        cfg["graph"] = graph_name
        all_tags.append(f"graph:{graph_name}")
    return wandb.init(
        entity=os.environ.get("WANDB_ENTITY", ENTITY),
        project=os.environ.get("WANDB_PROJECT", PROJECT),
        group=group or experiment,
        job_type=job_type,
        name=f"{experiment}-{machine}-{sha}",
        tags=all_tags,
        config=cfg,
    )


def log_eval(run, metrics: dict[str, float], step: int | None = None) -> None:
    """Log eval metrics under the 'eval/' prefix; also written to the run summary for easy comparison.

    Pass the same explicit ``step`` to log_eval and log_episodes: W&B drops a log whose step is lower than
    the current one, and a step-less log() advances the step."""
    payload = {f"eval/{k}": v for k, v in metrics.items()}
    run.log(payload, step=step)
    run.summary.update(payload)


def log_episodes(run, episodes: list[Episode], step: int = 0, name: str = "eval", table_rows: int = 200) -> None:
    """Log episodes two ways: a browsable wandb.Table (capped) and the full set as a JSONL artifact
    (what the webapp reads: complete paths and raw model text)."""
    cols = ["n_hops", "source", "target", "success", "end_reason", "optimal_steps", "extra_steps",
            "path_len", "n_invalid", "total_reward", "path"]
    table = wandb.Table(columns=cols)
    for e in episodes[:table_rows]:
        table.add_data(e.n_hops, e.source, e.target, e.success, e.end_reason, e.optimal_steps, e.extra_steps,
                       len(e.path) - 1, e.n_invalid, round(e.total_reward, 4), " -> ".join(e.path))
    run.log({f"{name}/episodes": table}, step=step)

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "episodes.jsonl"
        f.write_text("\n".join(json.dumps(e.to_dict()) for e in episodes))
        art = wandb.Artifact(f"episodes-{run.id}-{name}-step{step}", type="episodes",
                             metadata={"step": step, "n_episodes": len(episodes), "split": name})
        art.add_file(str(f), name="episodes.jsonl")
        logged = run.log_artifact(art)
        if not run.settings._offline and not run.settings.mode == "disabled":
            logged.wait()  # upload must finish before the temp dir is deleted; offline mode copies at log time
