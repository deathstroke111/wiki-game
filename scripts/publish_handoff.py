"""Package trained adapters + eval outputs for the dev machine (CUDA side), or verify a download (dev side).

  # CUDA box: see what would ship and its size, nothing written or uploaded
  uv run python scripts/publish_handoff.py --dry-run
  # CUDA box: write handoff/MANIFEST.txt (commit + push it) and upload the W&B artifact `wikigame-handoff`
  uv run python scripts/publish_handoff.py
  # dev machine: restore into the repo root, then verify against the manifest that arrived through git
  uv run wandb artifact get aiwizards/wiki-game/wikigame-handoff:latest --root .
  uv run python scripts/publish_handoff.py --verify .
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wikigame.handoff import build_manifest, default_files, read_manifest, verify, write_manifest

ARTIFACT = "wikigame-handoff"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."), help="repo root (file names in the manifest are relative to it)")
    ap.add_argument("--manifest", type=Path, default=Path("handoff/MANIFEST.txt"))
    ap.add_argument("--dry-run", action="store_true", help="list + hash the files; write and upload nothing")
    ap.add_argument("--no-upload", action="store_true", help="write the manifest but skip the W&B upload")
    ap.add_argument("--verify", metavar="DIR", type=Path, help="check DIR against the manifest and exit")
    a = ap.parse_args()

    if a.verify:
        entries = read_manifest(a.manifest)
        problems = verify(entries, a.verify)
        for p in problems:
            print("PROBLEM", p)
        print(f"{len(entries) - len(problems)}/{len(entries)} files OK" if problems else f"all {len(entries)} files verified OK")
        sys.exit(1 if problems else 0)

    files, warnings = default_files(a.root)
    for w in warnings:
        print("WARNING", w)
    entries = build_manifest(files, a.root)
    total = sum(s for _, s, _ in entries)
    for sha, size, rel in entries:
        print(f"{size / 2**20:9.1f} MB  {sha[:12]}  {rel}")
    print(f"{len(entries)} files, {total / 2**20:.0f} MB total")
    if a.dry_run:
        return

    write_manifest(entries, a.manifest, header="sha256<TAB>bytes<TAB>repo-relative path; see src/wikigame/handoff.py")
    print("wrote", a.manifest)
    if a.no_upload:
        return

    import wandb

    from wikigame.tracking import git_sha, init_run

    run = init_run("handoff-cuda", "handoff", {"n_files": len(entries), "total_mb": round(total / 2**20)}, tags=["handoff"])
    art = wandb.Artifact(ARTIFACT, type="model", metadata={"git_sha": git_sha(), "n_files": len(entries)},
                         description="LoRA adapters + eval outputs from the CUDA machine; file names are repo-relative paths")
    for _, _, rel in entries:
        art.add_file(str(a.root / rel), name=rel)
    logged = run.log_artifact(art)
    logged.wait()
    print(f"uploaded {ARTIFACT}:{logged.version}  ({run.url})")
    run.finish()


if __name__ == "__main__":
    main()
