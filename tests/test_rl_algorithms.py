"""The policy-gradient presets in wikigame.rl.algorithms: pure torch, no model."""
import pytest

torch = pytest.importorskip("torch")

from wikigame.rl.algorithms import (ALGORITHMS, Algorithm, advantages, aggregate, get_algorithm,  # noqa: E402
                                    per_token_loss, zero_std_groups)

R = torch.tensor([[1.0, 0.0, -1.0, 0.0], [2.0, 2.0, 2.0, 2.0]])


def test_baselines_and_scaling():
    grp = advantages(Algorithm("x", baseline="group_mean", scale_by_std=False), R)
    assert torch.allclose(grp[0], torch.tensor([1.0, 0.0, -1.0, 0.0])) and torch.all(grp[1] == 0)
    loo = advantages(get_algorithm("rloo"), R)
    # r_i - mean of the other three
    assert torch.allclose(loo[0], torch.tensor([1 - (-1 / 3), 0 - 0, -1 - (1 / 3), 0 - 0]))
    assert torch.allclose(loo, grp * 4 / 3)  # RLOO = G/(G-1) x the group-mean advantage
    batch = advantages(get_algorithm("reinforce"), R)
    assert torch.allclose(batch, R - R.mean())
    scaled = advantages(get_algorithm("grpo"), R)
    assert torch.allclose(scaled[0], grp[0] / (R[0].std() + 1e-4)) and torch.all(scaled[1] == 0)
    assert zero_std_groups(R).tolist() == [False, True]


def test_presets():
    assert set(ALGORITHMS) >= {"reinforce", "rloo", "grpo", "dapo", "cispo", "gspo", "trl-dapo"}
    d = get_algorithm("dapo")
    assert (d.eps_low, d.eps_high, d.agg, d.drop_zero_std) == (0.2, 0.28, "token_mean", True)
    assert get_algorithm("gspo").ratio_level == "sequence"
    assert get_algorithm("cispo").with_overrides(cispo_max=2.0, eps_low=None).cispo_max == 2.0
    with pytest.raises(ValueError):
        get_algorithm("ppo-but-typo")
    with pytest.raises(ValueError):
        Algorithm("bad", agg="sum")


def _grad(alg, logits, old, adv, mask, targets):
    logits = logits.clone().requires_grad_(True)
    logp = torch.log_softmax(logits, -1).gather(-1, targets[..., None]).squeeze(-1)
    loss, _ = per_token_loss(alg, logp, old if old is not None else logp.detach(), adv, mask)
    aggregate(alg, loss, mask, total_tokens=int(mask.sum()), total_seqs=mask.shape[0], max_completion=mask.shape[1]).backward()
    return logits.grad


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 5, 11, generator=g)
    targets = torch.randint(0, 11, (3, 5), generator=g)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)
    adv = torch.tensor([1.0, -0.5, 0.25])
    return logits, targets, mask, adv


def test_on_policy_pg_clip_cispo_and_gspo_share_one_gradient(batch):
    logits, targets, mask, adv = batch
    base = _grad(Algorithm("pg", objective="pg"), logits, None, adv, mask, targets)
    for alg in (Algorithm("c", objective="clip"), Algorithm("ci", objective="cispo")):
        assert torch.allclose(_grad(alg, logits, None, adv, mask, targets), base, atol=1e-6)
    seq_pg = _grad(Algorithm("p", objective="pg", agg="seq_mean"), logits, None, adv, mask, targets)
    gspo = _grad(get_algorithm("gspo"), logits, None, adv, mask, targets)
    assert torch.allclose(gspo, seq_pg, atol=1e-6)


def test_clip_stops_gradient_outside_the_range_but_cispo_does_not(batch):
    logits, targets, mask, adv = batch
    logp = torch.log_softmax(logits, -1).gather(-1, targets[..., None]).squeeze(-1)
    old = logp - 1.0  # every token's ratio is e (> 1.28): beyond the clip for positive advantages
    clip_g = _grad(get_algorithm("dapo"), logits, old, adv, mask, targets)
    cispo_g = _grad(get_algorithm("cispo"), logits, old, adv, mask, targets)
    assert torch.all(clip_g[0] == 0)  # positive advantage, ratio above 1 + eps_high: clipped
    assert clip_g[1].abs().sum() > 0  # negative advantage: min() keeps the unclipped term
    assert cispo_g[0].abs().sum() > 0  # CISPO keeps every token's gradient
    capped = _grad(Algorithm("ci", objective="cispo", cispo_max=2.0), logits, old, adv, mask, targets)
    full = _grad(Algorithm("ci", objective="cispo", cispo_max=10.0), logits, old, adv, mask, targets)
    assert torch.allclose(capped * torch.e, full * 2.0, atol=1e-5)  # weight min(e, 2) vs e
    _, st = per_token_loss(get_algorithm("dapo"), logp, old, adv, mask)
    assert st["clip_high_sum"] == mask[0].sum() + mask[2].sum()


def test_gspo_ratio_is_the_geometric_mean(batch):
    logits, targets, mask, adv = batch
    logp = torch.log_softmax(logits, -1).gather(-1, targets[..., None]).squeeze(-1)
    old = logp - torch.linspace(0, 0.4, 5)
    _, st = per_token_loss(get_algorithm("gspo"), logp, old, adv, mask)
    lr = ((logp - old) * mask).sum(1) / mask.sum(1)
    assert torch.isclose(st["ratio_sum"], (torch.exp(lr)[:, None] * mask).sum())


@pytest.mark.parametrize("agg", ["token_mean", "seq_mean", "seq_sum_const"])
def test_micro_batches_add_up_to_the_full_batch(batch, agg):
    logits, targets, mask, adv = batch
    alg = Algorithm("x", objective="pg", agg=agg)
    logp = torch.log_softmax(logits, -1).gather(-1, targets[..., None]).squeeze(-1)
    norm = dict(total_tokens=int(mask.sum()), total_seqs=3, max_completion=5)
    full = aggregate(alg, per_token_loss(alg, logp, logp.detach(), adv, mask)[0], mask, **norm)
    parts = sum(aggregate(alg, per_token_loss(alg, logp[s], logp[s].detach(), adv[s], mask[s])[0], mask[s], **norm)
                for s in (slice(0, 2), slice(2, 3)))
    assert torch.isclose(full, parts)
