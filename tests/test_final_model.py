"""Tests du prédicteur final (bundle multi-résolution)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from coenergy.ensemble import WeightedEnsemble  # noqa: E402
from coenergy.final_model import FinalModel, NoisyInput, predict_member  # noqa: E402
from coenergy.nets import build_hybrid_net, build_sequence_array  # noqa: E402
from coenergy.predict import Member  # noqa: E402
from coenergy.targets import TargetTransform  # noqa: E402

CFG = {
    "kind": "cnn_gru",
    "channels": 8,
    "n_conv": 1,
    "kernel": 3,
    "gru_hidden": 8,
    "gru_layers": 1,
    "tab_hidden": 8,
    "head_hidden": 8,
    "dropout": 0.0,
    "use_tab": False,
}
LABELS = ["HLC", "HLC_mit", "HLC_ext", "tau_b", "tau_n", "tau_inf"]


def _fake_data(n: int = 64, length: int = 480, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = 19 + rng.normal(0, 0.5, (n, length)).cumsum(axis=1) * 0.01
    u = (rng.random((n, length)) > 0.5) * 7000.0
    hb = rng.uniform(20, 120, (n, 2))
    tau = rng.uniform(5, 80, (n, 3))
    y = np.column_stack([hb.sum(axis=1), hb, tau])
    return t, u, y


def _member(name: str, factor: int, t, u, y) -> Member:
    tt = TargetTransform("A").fit(y)
    td = t[:, : (t.shape[1] // factor) * factor].reshape(len(t), -1, factor).mean(axis=2)
    ud = u[:, : (u.shape[1] // factor) * factor].reshape(len(u), -1, factor).mean(axis=2)
    _, stats = build_sequence_array(td, ud, np.arange(len(t)))
    torch.manual_seed(factor)
    net = build_hybrid_net(CFG, 0)
    return Member(name, "hybrid", tt, [], None, None, CFG, net.state_dict(), 0, stats)


def test_final_model_respects_hlc_constraint():
    t, u, y = _fake_data()
    members = [(2, _member("m2", 2, t, u, y)), (5, _member("m5", 5, t, u, y))]
    preds = {n: predict_member(m, f, t, u) for n, (f, m) in zip(["m2", "m5"], members, strict=True)}
    comb = WeightedEnsemble().fit(preds, y)
    model = FinalModel(LABELS, ["m2", "m5"], members, comb)
    df = model.predict(t, u)
    assert df.shape == (len(t), 6)
    assert np.all(np.isfinite(df.to_numpy()))
    np.testing.assert_allclose(df["HLC"], df["HLC_mit"] + df["HLC_ext"], rtol=1e-5)
    assert (df[["HLC_mit", "HLC_ext", "tau_b", "tau_n", "tau_inf"]] > 0).all().all()


def test_noisy_input_only_in_training():
    net = build_hybrid_net(CFG, 0)
    model = NoisyInput(net, sigma=0.5, std_t=1.0, std_dt=1.0)
    x = torch.zeros(4, 3, 96)
    model.eval()
    torch.testing.assert_close(model(x), net(x))
    model.train()
    torch.manual_seed(0)
    a = model(x)
    torch.manual_seed(1)
    b = model(x)
    assert not torch.allclose(a, b)
