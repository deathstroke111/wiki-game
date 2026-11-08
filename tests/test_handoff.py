import pytest

from wikigame.handoff import build_manifest, default_files, read_manifest, verify, write_manifest


def _tree(root):
    for rel, data in {
        "data/ckpt/grpo-step-r1-stage2/adapter_model.safetensors": b"final-weights",
        "data/ckpt/grpo-step-r1-stage2/adapter_config.json": b"{}",
        "data/ckpt/grpo-step-r1-stage2/wikigame_config.json": b"{}",
        "data/ckpt/grpo-step-r1-stage2/checkpoint-25/adapter_model.safetensors": b"step25",
        "data/ckpt/grpo-step-r1-stage2/checkpoint-25/adapter_config.json": b"{}",
        "data/ckpt/grpo-step-r1-stage2/checkpoint-100/adapter_model.safetensors": b"step100",
        "data/ckpt/grpo-step-r1-stage2/checkpoint-100/optimizer.pt": b"never shipped",
        "data/ckpt/grpo-step-r1-stage1/checkpoint-25/adapter_model.safetensors": b"lineage",
        "data/ckpt/grpo-step-r1-stage1/checkpoint-25/adapter_config.json": b"{}",
        "data/runs/eval-a.jsonl": b'{"ep": 1}\n',
        "data/logs/chain.log": b"chain",
        "data/logs/eval-a.log": b"eval log",
        "data/logs/evalchain.log": b"evalchain",
    }.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def _rels(files, root):
    return {f.relative_to(root).as_posix() for f in files}


def test_training_phase_ships_every_checkpoint_adapter_but_no_optimizer_or_evals(tmp_path):
    _tree(tmp_path)
    files, warnings = default_files(tmp_path, "training")
    rels = _rels(files, tmp_path)
    assert "data/ckpt/grpo-step-r1-stage2/adapter_model.safetensors" in rels
    assert "data/ckpt/grpo-step-r1-stage2/wikigame_config.json" in rels
    assert "data/ckpt/grpo-step-r1-stage2/checkpoint-25/adapter_model.safetensors" in rels
    assert "data/ckpt/grpo-step-r1-stage2/checkpoint-100/adapter_model.safetensors" in rels
    assert "data/ckpt/grpo-step-r1-stage1/checkpoint-25/adapter_model.safetensors" in rels  # lineage of stage 1b
    assert "data/logs/chain.log" in rels
    assert not any("optimizer" in r for r in rels)
    assert not any(r.startswith("data/runs/") or "eval" in r for r in rels)  # evals are a later phase
    assert any("grpo-step-r1-stage1b" in w for w in warnings)  # stage1b is absent here: reported, not fatal
    assert not any("grpo-step-r1-stage1/" in w for w in warnings)  # lineage dir has no final adapter and needs none


def test_eval_phase_ships_only_eval_outputs(tmp_path):
    _tree(tmp_path)
    rels = _rels(default_files(tmp_path, "eval")[0], tmp_path)
    assert rels == {"data/runs/eval-a.jsonl", "data/logs/eval-a.log", "data/logs/evalchain.log"}


def test_unknown_phase_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        default_files(tmp_path, "everything")


def test_manifest_roundtrip_and_verify_catches_damage(tmp_path):
    _tree(tmp_path)
    files, _ = default_files(tmp_path, "training")
    entries = build_manifest(files, tmp_path)
    m = tmp_path / "handoff" / "MANIFEST-training.txt"
    write_manifest(entries, m, header="test")
    assert read_manifest(m) == entries
    assert verify(entries, tmp_path) == []

    (tmp_path / "data/ckpt/grpo-step-r1-stage2/checkpoint-25/adapter_model.safetensors").write_bytes(b"STEP25")  # same size
    (tmp_path / "data/logs/chain.log").unlink()
    (tmp_path / "data/ckpt/grpo-step-r1-stage2/adapter_config.json").write_bytes(b"{ }")  # different size
    problems = verify(entries, tmp_path)
    assert any(p.startswith("sha256 mismatch") and "checkpoint-25" in p for p in problems)
    assert any(p.startswith("missing") and "chain.log" in p for p in problems)
    assert any(p.startswith("size mismatch") and "adapter_config" in p for p in problems)
