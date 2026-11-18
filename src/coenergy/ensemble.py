"""Combinaison de modèles : moyenne pondérée (poids positifs, somme 1) et stacking ridge.

Les prédictions sont combinées dans un espace cohérent avec la physique :
Hb0 et Hb1 en échelle linéaire (puis HLC = Hb0 + Hb1), constantes de temps en log.
Les poids sont appris sur la validation ; leur généralisation est estimée par validation
croisée **par motif d'excitation** sur la validation, avant l'évaluation finale sur le test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from coenergy.evaluation import group_kfold_indices


def to_space(y: np.ndarray) -> np.ndarray:
    """Indicateurs physiques (N, 6) → espace de combinaison (N, 5) : Hb0, Hb1, log τ."""
    return np.column_stack([y[:, 1], y[:, 2], np.log(np.clip(y[:, 3:6], 1e-6, None))])


def from_space(s: np.ndarray) -> np.ndarray:
    """Espace de combinaison → indicateurs physiques cohérents (HLC = Hb0 + Hb1)."""
    hb0, hb1 = np.clip(s[:, 0], 1e-3, None), np.clip(s[:, 1], 1e-3, None)
    return np.column_stack([hb0 + hb1, hb0, hb1, np.exp(s[:, 2:5])])


@dataclass
class WeightedEnsemble:
    """Moyenne pondérée par composante ; poids ≥ 0 normalisés à 1 (NNLS)."""

    names: list[str] = field(default_factory=list)
    weights: np.ndarray | None = None  # (n_models, 5)

    def fit(self, preds: dict[str, np.ndarray], y: np.ndarray) -> WeightedEnsemble:
        self.names = list(preds)
        target = to_space(y)
        stacked = np.stack([to_space(preds[n]) for n in self.names], axis=0)  # (M, N, 5)
        w = np.zeros((len(self.names), 5))
        for j in range(5):
            coef, _ = nnls(stacked[:, :, j].T, target[:, j])
            w[:, j] = coef / coef.sum() if coef.sum() > 0 else 1 / len(self.names)
        self.weights = w
        return self

    def predict(self, preds: dict[str, np.ndarray]) -> np.ndarray:
        assert self.weights is not None
        stacked = np.stack([to_space(preds[n]) for n in self.names], axis=0)
        return from_space(np.einsum("mnj,mj->nj", stacked, self.weights))


@dataclass
class RidgeStacker:
    """Stacking : une ridge par composante sur les prédictions de tous les modèles."""

    alpha: float = 1.0
    names: list[str] = field(default_factory=list)
    models: list = field(default_factory=list)

    def fit(self, preds: dict[str, np.ndarray], y: np.ndarray) -> RidgeStacker:
        from sklearn.linear_model import Ridge

        self.names = list(preds)
        x = np.concatenate([to_space(preds[n]) for n in self.names], axis=1)
        target = to_space(y)
        self.models = [Ridge(alpha=self.alpha).fit(x, target[:, j]) for j in range(5)]
        return self

    def predict(self, preds: dict[str, np.ndarray]) -> np.ndarray:
        x = np.concatenate([to_space(preds[n]) for n in self.names], axis=1)
        return from_space(np.column_stack([m.predict(x) for m in self.models]))


def cross_validated_predictions(
    make,
    preds_val: dict[str, np.ndarray],
    y_val: np.ndarray,
    groups_val: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Prédictions hors-pli d'un combinateur (`make()` renvoie un objet fit/predict), avec des plis
    par motif d'excitation : estime honnêtement le gain d'un ensemble avant de toucher au test."""
    out = np.zeros_like(y_val, dtype=np.float64)
    for tr, te in group_kfold_indices(groups_val, n_splits, seed):
        comb = make().fit({k: v[tr] for k, v in preds_val.items()}, y_val[tr])
        out[te] = comb.predict({k: v[te] for k, v in preds_val.items()})
    return out
