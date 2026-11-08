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
RUN_DIRS = ("grpo-step-r1-stage1b", "grpo-step-r1-stage2")  # under data/ckpt
FINAL_EXTRA = ("wikigame_config.json",)  # training config, written next to a finished run's adapter
KEEP_CHECKPOINTS = {"grpo-step-r1-stage1b": (100, 200), "grpo-step-r1-stage2": (100, 200)}
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


def default_files(root: Path) -> tuple[list[Path], list[str]]:
    """What to ship: final adapters (+ training config), a few intermediate adapters (adapter files only, no optimizer
    state), eval episode JSONL, and the training/eval logs. Returns (existing files, warnings for missing ones)."""
    ckpt, files, warnings = root / "data" / "ckpt", [], []

    def add(p: Path) -> None:
        (files.append(p) if p.is_file() else warnings.append(f"missing: {p.relative_to(root).as_posix()}"))

    for run in RUN_DIRS:
        for name in ADAPTER_FILES + FINAL_EXTRA:
            add(ckpt / run / name)
        for step in KEEP_CHECKPOINTS.get(run, ()):
            for name in ADAPTER_FILES:
                add(ckpt / run / f"checkpoint-{step}" / name)
    files += sorted((root / "data" / "runs").glob("*.jsonl"))
    for log in sorted((root / "data" / "logs").glob("*.log")):
        if log.stat().st_size <= MAX_LOG_BYTES:
            files.append(log)
        else:
            warnings.append(f"skipped (> {MAX_LOG_BYTES // 2**20} MB): {log.relative_to(root).as_posix()}")
    return files, warnings
