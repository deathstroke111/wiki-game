"""Hand trained adapters + eval outputs from the CUDA machine to the dev machine, verifiably.

The CUDA side hashes the files it is about to ship and writes ``handoff/MANIFEST.txt`` (small, committed to git);
the bytes travel as a W&B artifact. The dev side downloads the artifact into the repo root (artifact file names are
repo-relative paths) and runs ``verify`` against the manifest it got through git, so a truncated or wrong download
is caught. Pure stdlib so the Mac can verify without any GPU stack.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
PHASES = ("training", "eval")  # shipped separately: training outputs are final early, evals take hours
RUN_DIRS = ("grpo-step-r1-stage1b", "grpo-step-r1-stage2")  # finished runs under data/ckpt (final adapter + config)
LINEAGE_DIRS = ("grpo-step-r1-stage1",)  # killed first run: only its checkpoint-25 matters (stage 1b resumed from it)
FINAL_EXTRA = ("wikigame_config.json",)  # training config, written next to a finished run's adapter
MAX_LOG_BYTES = 20 * 2**20


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


Entry = tuple[str, int, str]  # (sha256, size_bytes, repo-relative posix path)


def build_manifest(files: list[Path], root: Path) -> list[Entry]:
    root = root.resolve()
    return [(sha256_file(f), f.stat().st_size, f.resolve().relative_to(root).as_posix()) for f in sorted(files)]


def write_manifest(entries: list[Entry], path: Path, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {ln}" for ln in header.splitlines()] + [f"{sha}\t{size}\t{rel}" for sha, size, rel in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[Entry]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            sha, size, rel = line.split("\t")
            entries.append((sha, int(size), rel))
    return entries


def verify(entries: list[Entry], root: Path) -> list[str]:
    """Problems found (empty list = every file present with the right size and sha256)."""
    problems = []
    for sha, size, rel in entries:
        f = root / rel
        if not f.is_file():
            problems.append(f"missing: {rel}")
        elif f.stat().st_size != size:
            problems.append(f"size mismatch: {rel} ({f.stat().st_size} != {size})")
        elif sha256_file(f) != sha:
            problems.append(f"sha256 mismatch: {rel}")
    return problems


def _small_logs(root: Path, patterns: tuple[str, ...], files: list[Path], warnings: list[str]) -> None:
    for pat in patterns:
        for log in sorted((root / "data" / "logs").glob(pat)):
            if log.stat().st_size <= MAX_LOG_BYTES:
                files.append(log)
            else:
                warnings.append(f"skipped (> {MAX_LOG_BYTES // 2**20} MB): {log.relative_to(root).as_posix()}")


def default_files(root: Path, phase: str = "training") -> tuple[list[Path], list[str]]:
    """What to ship in each phase. Returns (existing files, warnings for missing/skipped ones).

    training: every checkpoint's adapter files (no optimizer state) + the final adapter and training config of each
              finished run, and the small training logs. All of it is immutable once training has finished.
    eval:     eval episode JSONL (data/runs) and the eval logs. Send only after the eval chain has finished."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    ckpt, files, warnings = root / "data" / "ckpt", [], []

    def add(p: Path) -> None:
        (files.append(p) if p.is_file() else warnings.append(f"missing: {p.relative_to(root).as_posix()}"))

    if phase == "training":
        for run in RUN_DIRS + LINEAGE_DIRS:
            if run in RUN_DIRS:
                for name in ADAPTER_FILES + FINAL_EXTRA:
                    add(ckpt / run / name)
            for cp in sorted((ckpt / run).glob("checkpoint-*")):
                for name in ADAPTER_FILES:
                    add(cp / name)
        _small_logs(root, ("chain.log", "stage*.log"), files, warnings)
    else:
        files += sorted((root / "data" / "runs").glob("*.jsonl"))
        _small_logs(root, ("eval*.log",), files, warnings)
    return files, warnings
