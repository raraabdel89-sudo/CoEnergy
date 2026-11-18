"""Métriques de régression par indicateur, par famille et par sous-groupe."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def family_of(label: str) -> str:
    """Famille d'un indicateur : "HLC", "Hb" ou "CDT"."""
    if label == "HLC":
        return "HLC"
    if label.startswith("tau"):
        return "CDT"
    return "Hb"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Métriques d'une cible scalaire.

    Returns:
        rmse, mae, p95_abs (95e percentile de l'erreur absolue, comme dans le paper),
        bias (erreur moyenne signée), mape (%), p95_rel (%) et r2.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    err = np.asarray(y_pred, dtype=np.float64) - y_true
    abs_err = np.abs(err)
    rel = abs_err / np.maximum(np.abs(y_true), 1e-9)
    ss_res = float((err**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(abs_err.mean()),
        "p95_abs": float(np.percentile(abs_err, 95)),
        "bias": float(err.mean()),
        "mape_%": float(100 * rel.mean()),
        "p95_rel_%": float(100 * np.percentile(rel, 95)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def metrics_table(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    """Tableau des métriques, une ligne par indicateur, avec sa famille."""
    rows = []
    for k, label in enumerate(labels):
        rows.append(
            {
                "famille": family_of(label),
                "indicateur": label,
                **regression_metrics(y_true[:, k], y_pred[:, k]),
            }
        )
    return pd.DataFrame(rows).set_index(["famille", "indicateur"])


def compare_models(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    labels: Sequence[str],
    metric: str = "rmse",
) -> pd.DataFrame:
    """Compare plusieurs modèles sur une métrique (lignes : indicateurs, colonnes : modèles)."""
    cols = {name: metrics_table(y_true, pred, labels)[metric] for name, pred in predictions.items()}
    return pd.DataFrame(cols)


def metrics_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    groups: np.ndarray,
    metric: str = "mape_%",
) -> pd.DataFrame:
    """Une métrique calculée séparément pour chaque groupe (ex. type d'excitation, décile)."""
    groups = np.asarray(groups)
    out = {}
    for g in pd.unique(groups):
        mask = groups == g
        out[g] = metrics_table(y_true[mask], y_pred[mask], labels)[metric]
    return pd.DataFrame(out)


def constraint_violation(hlc: np.ndarray, hb: np.ndarray) -> np.ndarray:
    """Écart absolu |HLC - Σ Hb|, en W/K (doit être ~0 dans les données et les prédictions)."""
    return np.abs(np.asarray(hlc) - np.asarray(hb).sum(axis=1))
