"""Tests unitaires sur des données synthétiques (pas besoin du vrai jeu de données)."""

from __future__ import annotations

import numpy as np
import torch

from coenergy.config import ModelConfig, NoiseConfig, TrainConfig
from coenergy.data import (
    BatchIterator,
    InputScaler,
    SimulationData,
    TargetScaler,
    make_splits,
)
from coenergy.features import excitation_descriptors
from coenergy.losses import multitask_loss
from coenergy.metrics import constraint_violation, metrics_table
from coenergy.model import CNNGRU
from coenergy.train import predict


def fake_data(n: int = 64, t: int = 120) -> SimulationData:
    """Jeu factice respectant les formes et la contrainte HLC = somme(Hb)."""
    gen = torch.Generator().manual_seed(0)
    hb = torch.rand(n, 2, generator=gen) * 50 + 5
    return SimulationData(
        temperature=20 + torch.randn(n, t, generator=gen).cumsum(1) * 0.01,
        excitation=(torch.rand(n, t, generator=gen) > 0.5).float() * 2000,
        hlc=hb.sum(1),
        hb=hb,
        tau=torch.rand(n, 3, generator=gen) * 100 + 1,
        hb_labels=["HLC_ext", "HLC_mit"],
        tau_labels=["tau_b", "tau_n", "tau_inf"],
    )


def test_splits_are_disjoint_and_complete() -> None:
    s = make_splits(100, 0.1, 0.2, seed=1)
    allidx = torch.cat(list(s.values()))
    assert len(allidx) == 100 and len(allidx.unique()) == 100


def test_target_scaler_roundtrip() -> None:
    data = fake_data()
    ts = TargetScaler.fit(data, torch.arange(len(data)))
    assert torch.allclose(ts.decode_tau(ts.encode_tau(data.tau)), data.tau, rtol=1e-4)
    assert torch.allclose(ts.decode_hb(ts.encode_hb(data.hb)), data.hb, rtol=1e-5)


def test_model_shapes_and_constraint() -> None:
    data = fake_data()
    idx = torch.arange(len(data))
    it = BatchIterator(
        data,
        idx,
        InputScaler.fit(data, idx),
        TargetScaler.fit(data, idx),
        batch_size=16,
        noise=NoiseConfig(),
    )
    model = CNNGRU(ModelConfig(conv_channels=(8, 8), hidden_size=16))
    batch = next(iter(it))
    out = model(batch.x)
    assert out["hb"].shape == (16, 2) and out["tau_z"].shape == (16, 3)
    assert torch.all(out["hb"] >= 0)
    assert torch.allclose(out["hlc"], out["hb"].sum(1))
    loss, parts = multitask_loss(out, batch, it.target_scaler, TrainConfig().loss_weights)
    loss.backward()
    assert set(parts) == {"hb", "hlc", "tau"}

    preds = predict(model, it, it.target_scaler)
    assert np.allclose(constraint_violation(preds.hlc, preds.hb), 0, atol=1e-3)


def test_metrics_and_descriptors() -> None:
    y = np.random.default_rng(0).uniform(1, 10, size=(50, 6))
    labels = ["HLC", "HLC_ext", "HLC_mit", "tau_b", "tau_n", "tau_inf"]
    table = metrics_table(y, y, labels)
    assert np.allclose(table["rmse"], 0)
    u = np.zeros((2, 100))
    u[0, ::10] = 2000.0  # PRBS-like
    u[1] = np.clip(1500 * np.sin(np.linspace(0, 12, 100)), 0, 2000)  # sinus
    kinds = excitation_descriptors(u)["kind"].tolist()
    assert kinds == ["PRBS", "sinus"]
