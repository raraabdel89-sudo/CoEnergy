"""Métriques, journal d'expériences et découpages pour la phase de modélisation."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Découpage par motif d'excitation
# --------------------------------------------------------------------------------------
def make_group_splits(
    groups: np.ndarray,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    force_train: Sequence[int] = (),
) -> dict[str, np.ndarray]:
    """Répartit des groupes (motifs U) entre train / val / test ; un groupe n'est que dans un
    ensemble. Les groupes de `force_train` vont au train. Renvoie des indices de simulations."""
    rng = np.random.default_rng(seed)
    candidates = rng.permutation(np.setdiff1d(np.unique(groups), list(force_train)))
    n_test = int(round(len(candidates) * test_frac))
    n_val = int(round(len(candidates) * val_frac))
    test_g, val_g = candidates[:n_test], candidates[n_test : n_test + n_val]
    in_test, in_val = np.isin(groups, test_g), np.isin(groups, val_g)
    return {
        "train": np.where(~in_test & ~in_val)[0],
        "val": np.where(in_val)[0],
        "test": np.where(in_test)[0],
    }


def save_splits(path: str | Path, splits: dict[str, np.ndarray]) -> None:
    """Sauvegarde les indices de découpage (pour réutilisation exacte dans le rapport)."""
    np.savez(path, **{k: np.asarray(v) for k, v in splits.items()})


def load_splits(path: str | Path) -> dict[str, np.ndarray]:
    """Recharge des indices sauvegardés par `save_splits`."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def group_kfold_indices(
    groups: np.ndarray, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold par groupe (aucun motif partagé entre plis)."""
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    folds = np.array_split(uniq, n_splits)
    out = []
    for f in folds:
        te = np.isin(groups, f)
        out.append((np.where(~te)[0], np.where(te)[0]))
    return out


# --------------------------------------------------------------------------------------
# Métriques
# --------------------------------------------------------------------------------------
def indicator_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]
) -> pd.DataFrame:
    """R², RMSE, MAE, MSE, RMSE normalisé (RMSE / écart-type) et biais, par indicateur."""
    rows = {}
    for j, lab in enumerate(labels):
        yt = np.asarray(y_true[:, j], dtype=np.float64)
        err = np.asarray(y_pred[:, j], dtype=np.float64) - yt
        mse = float((err**2).mean())
        var = float(yt.var())
        rows[lab] = {
            "r2": 1 - mse / var if var > 0 else np.nan,
            "rmse": np.sqrt(mse),
            "mae": float(np.abs(err).mean()),
            "mse": mse,
            "nrmse": np.sqrt(mse / var) if var > 0 else np.nan,
            "p95_abs": float(np.percentile(np.abs(err), 95)),
            "mape_%": float(100 * np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-9))),
            "bias": float(err.mean()),
        }
    return pd.DataFrame(rows).T


def summary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str], prefix: str
) -> dict:
    """Résumé plat : R² moyen, R² par indicateur, RMSE/MAE/MSE du HLC, RMSE normalisé moyen."""
    m = indicator_metrics(y_true, y_pred, labels)
    out = {
        f"r2_{prefix}_mean": float(m["r2"].mean()),
        f"nrmse_{prefix}_mean": float(m["nrmse"].mean()),
    }
    for lab in labels:
        out[f"r2_{prefix}_{lab}"] = float(m.loc[lab, "r2"])
    out[f"rmse_{prefix}_HLC"] = float(m.loc[labels[0], "rmse"])
    out[f"mae_{prefix}_HLC"] = float(m.loc[labels[0], "mae"])
    out[f"mse_{prefix}_HLC"] = float(m.loc[labels[0], "mse"])
    return out


def segment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    segments: Sequence,
    metric: str = "r2",
) -> pd.DataFrame:
    """Une métrique par segment (lignes) et par indicateur (colonnes), avec l'effectif."""
    seg = np.asarray(segments)
    rows = {}
    for s in pd.unique(seg):
        mask = seg == s
        if mask.sum() < 20:
            continue
        row = indicator_metrics(y_true[mask], y_pred[mask], labels)[metric].to_dict()
        row["effectif"] = int(mask.sum())
        rows[s] = row
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------------------
# Journal d'expériences
# --------------------------------------------------------------------------------------
class ExperimentLog:
    """Enregistre chaque expérience (métriques train/val, hyperparamètres, durée) dans un CSV,
    réécrit après chaque ajout pour survivre à une interruption de session."""

    def __init__(self, path: str | Path, labels: Sequence[str]) -> None:
        self.path = Path(path)
        self.labels = list(labels)
        self.rows: list[dict] = []
        if self.path.exists():
            self.rows = pd.read_csv(self.path).to_dict("records")

    def add(
        self,
        name: str,
        family: str,
        params: dict,
        y_train: np.ndarray,
        pred_train: np.ndarray,
        y_val: np.ndarray,
        pred_val: np.ndarray,
        seconds: float,
        **extra: object,
    ) -> dict:
        """Ajoute une expérience et renvoie sa ligne de résultats."""
        row = {
            "nom": name,
            "famille": family,
            "params": json.dumps(params, default=str),
            "temps_s": round(seconds, 1),
        }
        row.update(summary_metrics(y_train, pred_train, self.labels, "train"))
        row.update(summary_metrics(y_val, pred_val, self.labels, "val"))
        row["ecart_r2_train_val"] = row["r2_train_mean"] - row["r2_val_mean"]
        row.update(extra)
        self.rows = [r for r in self.rows if r["nom"] != name] + [row]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
        print(
            f"[{name}] R² val moyen = {row['r2_val_mean']:.4f}"
            f" | HLC = {row[f'r2_val_{self.labels[0]}']:.4f}"
            f" | train = {row['r2_train_mean']:.4f} | {seconds:.0f} s"
        )
        return row

    def table(self, sort_by: str = "r2_val_mean") -> pd.DataFrame:
        """Tableau comparatif trié."""
        if not self.rows:
            return pd.DataFrame()
        cols = [
            "nom",
            "famille",
            "r2_train_mean",
            "r2_val_mean",
            "ecart_r2_train_val",
            *[f"r2_val_{lab}" for lab in self.labels],
            f"rmse_val_{self.labels[0]}",
            f"mae_val_{self.labels[0]}",
            f"mse_val_{self.labels[0]}",
            "nrmse_val_mean",
            "temps_s",
            "params",
        ]
        df = pd.DataFrame(self.rows)
        cols = [c for c in cols if c in df.columns]
        return df[cols].sort_values(sort_by, ascending=False).reset_index(drop=True)


class Timer:
    """Chronomètre simple : `with Timer() as t: ...; t.seconds`."""

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start
