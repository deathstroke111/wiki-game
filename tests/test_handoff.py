from wikigame.handoff import build_manifest, default_files, read_manifest, verify, write_manifest


def _tree(root):
    for rel, data in {
        "data/ckpt/grpo-step-r1-stage2/adapter_model.safetensors": b"final-weights",
        "data/ckpt/grpo-step-r1-stage2/adapter_config.json": b"{}",
        "data/ckpt/grpo-step-r1-stage2/wikigame_config.json": b"{}",
        "data/ckpt/grpo-step-r1-stage2/checkpoint-100/adapter_model.safetensors": b"step100",
        "data/ckpt/grpo-step-r1-stage2/checkpoint-100/optimizer.pt": b"never shipped",
        "data/runs/eval-a.jsonl": b'{"ep": 1}\n',
        "data/logs/stage2.log": b"log",
    }.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def test_default_files_picks_adapters_not_optimizer_state(tmp_path):
    _tree(tmp_path)
    files, warnings = default_files(tmp_path)
    rels = {f.relative_to(tmp_path).as_posix() for f in files}
    assert "data/ckpt/grpo-step-r1-stage2/adapter_model.safetensors" in rels
    assert "data/ckpt/grpo-step-r1-stage2/checkpoint-100/adapter_model.safetensors" in rels
    assert "data/runs/eval-a.jsonl" in rels and "data/logs/stage2.log" in rels
    assert not any("optimizer" in r for r in rels)
    assert any("grpo-step-r1-stage1b" in w for w in warnings)  # stage1b is absent: reported, not fatal


def test_manifest_roundtrip_and_verify_catches_damage(tmp_path):
    _tree(tmp_path)
    files, _ = default_files(tmp_path)
    entries = build_manifest(files, tmp_path)
    m = tmp_path / "handoff" / "MANIFEST.txt"
    write_manifest(entries, m, header="test")
    assert read_manifest(m) == entries
    assert verify(entries, tmp_path) == []

    (tmp_path / "data/runs/eval-a.jsonl").write_bytes(b'{"ep": 2}\n')  # same size, different bytes
    (tmp_path / "data/logs/stage2.log").unlink()
    (tmp_path / "data/ckpt/grpo-step-r1-stage2/adapter_config.json").write_bytes(b"{ }")  # different size
    problems = verify(entries, tmp_path)
    assert any(p.startswith("sha256 mismatch") and "eval-a" in p for p in problems)
    assert any(p.startswith("missing") and "stage2.log" in p for p in problems)
    assert any(p.startswith("size mismatch") and "adapter_config" in p for p in problems)
