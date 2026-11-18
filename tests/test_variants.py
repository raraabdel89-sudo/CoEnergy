"""Tests des variantes de l'annexe A3 (réponse moyenne de la population, GRU bidirectionnel)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from coenergy.nets import HybridNet  # noqa: E402
from coenergy.variants import (  # noqa: E402
    BiGRUEncoder,
    fit_population_response,
    population_prediction,
    standardize_channels,
)


def test_population_response_recovers_known_system():
    rng = np.random.default_rng(0)
    n, length = 2000, 48
    u = (rng.random((n, length)) > 0.5).astype(float) * 7.0
    h_true = 0.3 * np.exp(-np.arange(length) / 6.0)
    f_true = -0.01 * np.arange(length)
    dt = population_prediction(u, h_true, f_true) + rng.normal(0, 0.01, (n, length))
    h, f = fit_population_response(dt, u, np.arange(n), n_sub=n, ridge=1e-8)
    np.testing.assert_allclose(h[:10], h_true[:10], atol=0.01)
    np.testing.assert_allclose(f, f_true, atol=0.05)


def test_standardize_uses_train_statistics():
    rng = np.random.default_rng(1)
    a, b = rng.normal(5, 2, (100, 20)), rng.normal(-1, 3, (100, 20))
    x = standardize_channels([a, b], np.arange(50))
    assert x.shape == (100, 2, 20)
    np.testing.assert_allclose(x[:50].mean(axis=(0, 2)), 0, atol=1e-5)


def test_bigru_encoder_shapes_and_hybrid_head():
    enc = BiGRUEncoder(3, channels=8, n_conv=2, kernel=3, hidden=16, layers=2)
    x = torch.randn(4, 3, 96)
    assert enc(x).shape == (4, enc.out_features) == (4, 64)
    net = HybridNet(enc, 0, 8, 8, 5, 0.0)
    assert net(x).shape == (4, 5)
