import torch

from cltllm.metrics import sliced_wasserstein, mmd_rff, energy_distance, js_divergence_logits
from cltllm.cacheops import mix_legacy_cache
from cltllm.governance import LowRankGovernance


def test_distribution_metrics_zero_on_identity():
    x = torch.randn(16, 32)
    assert sliced_wasserstein(x, x.clone(), 16, 1) < 1e-6
    assert mmd_rff(x, x.clone(), 32, 1) < 1e-6
    assert energy_distance(x, x.clone()) < 1e-6
    assert js_divergence_logits(torch.randn(20), torch.zeros(20)) >= 0


def test_cache_mixing():
    a = tuple((torch.ones(1, 1, 2, 2) * i, torch.ones(1, 1, 2, 2) * (i + 10)) for i in range(4))
    b = tuple((torch.ones(1, 1, 2, 2) * -i, torch.ones(1, 1, 2, 2) * -(i + 10)) for i in range(4))
    m = mix_legacy_cache(a, b, {1, 3})
    assert torch.equal(m[1][0], a[1][0])
    assert torch.equal(m[0][0], b[0][0])


def test_governance_snapshot_is_exact():
    dev = torch.device("cpu")
    g = LowRankGovernance(32, 4, 2, 0.02, 0.05, 1.0, dev, seed=5)
    g.reset(2)
    h = torch.randn(2, 32)
    g.update(h, torch.tensor([1.0, -1.0]))
    snap = g.snapshot_cpu()
    g2 = LowRankGovernance.from_snapshot(snap, dev)
    assert torch.equal(g.state, g2.state)
    assert torch.equal(g.A, g2.A)
    assert torch.equal(g.B, g2.B)
