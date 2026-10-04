"""Project fake-quant math and gradient tests; no exporter or upstream C."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from train import qat4


def test_zero_rows_groups_and_rounding_match_scalar_contract():
    rng = np.random.default_rng(23)
    w = rng.standard_normal((8, 64)).astype(np.float32)
    w[0] = 0
    w[1, 32:] = 0
    expected = np.empty_like(w)
    for row in range(len(w)):
        peak = float(np.abs(w[row]).max())
        for start in range(0, 64, 32):
            group = w[row, start:start + 32]
            scale = int(float(np.abs(group).max()) / peak * 32767 + 0.5) if peak else 0
            effective = (peak / 7) * scale / 32768
            for col, value in enumerate(group):
                x = float(value) / effective if effective else 0
                q = int(x + (0.5 if x >= 0 else -0.5))
                expected[row, start + col] = min(7, max(-8, q)) * effective
    actual = qat4.WQAT4.apply(torch.from_numpy(w), 32).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert np.isfinite(actual).all()
    assert (actual[0] == 0).all() and (actual[1, 32:] == 0).all()


def test_ste_gradient_is_identity():
    w = torch.randn(4, 64, requires_grad=True)
    qat4.WQAT4.apply(w, 32).sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))


def test_non_group_aligned_shape_is_unchanged():
    w = torch.randn(3, 17)
    assert torch.equal(qat4.WQAT4.apply(w, 32), w)
