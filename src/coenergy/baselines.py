"""Modèles de référence, pour juger l'apport réel du CNN-GRU.

- `mean_baseline` : prédit la moyenne du train. C'est le score à battre absolument.
- `RidgeBaseline` : régression linéaire régularisée (forme fermée, en PyTorch) sur des
  descripteurs physiques simples des séries.
"""

from __future__ import annotations

import numpy as np
import torch

from coenergy.data import SimulationData, TargetScaler
from coenergy.features import summary_features
from coenergy.train import Predictions


def mean_baseline(
    data: SimulationData, train_idx: torch.Tensor, eval_idx: torch.Tensor
) -> Predictions:
    """Prédit, pour chaque indicateur, sa moyenne sur le train (moyenne géométrique pour τ)."""
    n = len(eval_idx)
    hb_mean = data.hb[train_idx].mean(dim=0)
    tau_mean = torch.exp(torch.log(data.tau[train_idx].clamp_min(1e-6)).mean(dim=0))
    return Predictions(
        index=eval_idx.numpy(),
        hlc=np.full(n, float(hb_mean.sum())),
        hb=np.tile(hb_mean.numpy(), (n, 1)),
        tau=np.tile(tau_mean.numpy(), (n, 1)),
    )


def compute_features(
    data: SimulationData, indices: torch.Tensor, n_points: int = 24, chunk: int = 50_000
) -> torch.Tensor:
    """Calcule les descripteurs `summary_features` par morceaux. Renvoie (N, F) en float64."""
    parts = []
    for start in range(0, len(indices), chunk):
        idx = indices[start : start + chunk]
        feats = summary_features(
            data.temperature[idx].numpy(), data.excitation[idx].numpy(), n_points=n_points
        )
        parts.append(torch.from_numpy(feats))
    return torch.cat(parts)


class RidgeBaseline:
    """Régression ridge multi-sorties en forme fermée : W = (XᵀX + αI)⁻¹ XᵀY."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.x_mean: torch.Tensor | None = None
        self.x_std: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None

    def _design(self, x: torch.Tensor) -> torch.Tensor:
        assert self.x_mean is not None and self.x_std is not None
        z = (x - self.x_mean) / self.x_std
        return torch.cat([z, torch.ones(z.shape[0], 1, dtype=z.dtype)], dim=1)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> RidgeBaseline:
        """Ajuste le modèle. x : (N, F), y : (N, K)."""
        self.x_mean = x.mean(dim=0)
        self.x_std = x.std(dim=0).clamp_min(1e-9)
        a = self._design(x)
        reg = self.alpha * torch.eye(a.shape[1], dtype=a.dtype)
        reg[-1, -1] = 0.0  # pas de régularisation sur le biais
        self.weights = torch.linalg.solve(a.T @ a + reg, a.T @ y)
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Prédit (N, K)."""
        assert self.weights is not None, "Appeler fit() d'abord"
        return self._design(x) @ self.weights


def ridge_baseline(
    data: SimulationData,
    train_idx: torch.Tensor,
    eval_idx: torch.Tensor,
    target_scaler: TargetScaler,
    alpha: float = 1.0,
) -> Predictions:
    """Entraîne la ridge sur le train (cibles normalisées) et prédit sur `eval_idx`."""

    def targets(idx: torch.Tensor) -> torch.Tensor:
        hb = target_scaler.encode_hb(data.hb[idx])
        tau = target_scaler.encode_tau(data.tau[idx])
        return torch.cat([hb, tau], dim=1).double()

    model = RidgeBaseline(alpha).fit(compute_features(data, train_idx), targets(train_idx))
    out = model.predict(compute_features(data, eval_idx)).float()
    n_hb = data.hb.shape[1]
    hb = target_scaler.decode_hb(out[:, :n_hb].clamp_min(0.0))
    tau = target_scaler.decode_tau(out[:, n_hb:])
    return Predictions(
        index=eval_idx.numpy(), hlc=hb.sum(dim=1).numpy(), hb=hb.numpy(), tau=tau.numpy()
    )
