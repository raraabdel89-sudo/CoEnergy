"""Tests de la chaîne de modélisation tabulaire (sans GPU)."""

from __future__ import annotations

import numpy as np

from coenergy.ensemble import WeightedEnsemble, cross_validated_predictions
from coenergy.evaluation import make_group_splits
from coenergy.fe import TabularPreprocessor, arx_features, build_features, select_columns
from coenergy.predict import Member, Predictor
from coenergy.targets import TargetTransform


class _LinearModel:
    """Moindres carrés avec biais (numpy) : estimateur minimal exposant `.predict`."""

    def __init__(self, x: np.ndarray, z: np.ndarray) -> None:
        xb = np.column_stack([x, np.ones(len(x))])
        self.coef = np.linalg.lstsq(xb, z, rcond=None)[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([x, np.ones(len(x))]) @ self.coef


def _toy(n: int = 400, length: int = 96, seed: int = 0):
    """Système du 1er ordre : T_{k+1} = a T_k + b U_k, avec HLC ∝ (1 - a) / b."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.90, 0.99, n)
    b = rng.uniform(5e-5, 2e-4, n)
    u = (rng.random((n, length)) > 0.5) * 7000.0
    t = np.zeros((n, length))
    t[:, 0] = 24.62
    for k in range(1, length):
        t[:, k] = 24.62 + a * (t[:, k - 1] - 24.62) + b * u[:, k - 1]
    hlc = 1000 * (1 - a) / (b * 1000)
    hb0 = hlc * rng.uniform(0.3, 0.7, n)
    tau = np.column_stack([rng.uniform(20, 80, n), rng.uniform(5, 15, n), rng.uniform(15, 50, n)])
    y = np.column_stack([hlc, hb0, hlc - hb0, tau])
    return t, u, y, rng.integers(0, 40, n)


def test_arx_recovers_static_gain() -> None:
    t, u, y, _ = _toy()
    f = arx_features(t, u, dt_minutes=5.0, order=1)
    assert np.corrcoef(np.log(f["arx1_H_W_per_K"]), np.log(y[:, 0]))[0, 1] > 0.99


def test_target_transforms_roundtrip_and_constraint() -> None:
    _, _, y, _ = _toy()
    for param in "AB":
        tt = TargetTransform(param).fit(y)
        back = tt.decode(tt.encode(y))
        assert np.allclose(back, y, rtol=1e-4)
        assert np.allclose(back[:, 0], back[:, 1] + back[:, 2])


def test_preprocessor_handles_nan_and_scalings() -> None:
    t, u, _, _ = _toy()
    u[:20] = 0.0  # simulations sans chauffe → descripteurs NaN
    feats = build_features(t, u, 5.0, sets=("phys", "arx2"))
    assert feats.isna().any().any()
    for scaling in ("standard", "robust", "quantile"):
        x = TabularPreprocessor(scaling).fit(feats).transform(feats)
        assert np.isfinite(x).all()


def test_group_splits_disjoint() -> None:
    groups = np.random.default_rng(0).integers(0, 50, 1000)
    s = make_group_splits(groups, 0.2, 0.2, seed=1, force_train=[0])
    assert not set(groups[s["train"]]) & set(groups[s["test"]])
    assert 0 not in set(groups[s["test"]]) | set(groups[s["val"]])
    assert len(np.concatenate(list(s.values()))) == 1000


def test_ensembles_and_predictor_roundtrip(tmp_path) -> None:
    t, u, y, groups = _toy()
    feats = build_features(t, u, 5.0, sets=("phys", "arx1"))
    cols = select_columns(feats, ["phys", "arx1"])
    pp = TabularPreprocessor().fit(feats[cols])
    x = pp.transform(feats[cols])
    tt = TargetTransform("A").fit(y)
    model = _LinearModel(x, tt.encode(y))
    p1 = tt.decode(model.predict(x))
    p2 = p1 * 1.01
    oof = cross_validated_predictions(WeightedEnsemble, {"a": p1, "b": p2}, y, groups, n_splits=4)
    assert oof.shape == y.shape and np.allclose(oof[:, 0], oof[:, 1] + oof[:, 2])
    pred = Predictor(
        ["HLC", "m", "e", "tb", "tn", "ti"],
        1,
        5.0,
        ["phys", "arx1"],
        members=[Member("lineaire", "sklearn", tt, cols, pp, model)],
    )
    pred.save(tmp_path / "p.joblib")
    out = Predictor.load(tmp_path / "p.joblib").predict(t, u)
    assert np.allclose(out.to_numpy(), p1, rtol=1e-4)
