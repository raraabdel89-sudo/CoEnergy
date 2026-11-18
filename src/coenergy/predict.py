"""Prédicteur final autonome : chaîne complète séries brutes → indicateurs physiques.

Un `Predictor` regroupe un ou plusieurs membres (GBM / modèle scikit-learn, réseau tabulaire,
réseau hybride) et, éventuellement, une pondération d'ensemble. Il se sauvegarde dans un
fichier unique (joblib) et s'utilise ainsi :

    from coenergy.predict import Predictor
    pred = Predictor.load("artifacts/best_model.joblib")
    df = pred.predict(temperature, excitation, input_dt_minutes=1.0)   # T et U bruts (N, 480)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from coenergy.fe import TabularPreprocessor, build_features
from coenergy.targets import TargetTransform


def downsample_series(x: np.ndarray, factor: int) -> np.ndarray:
    """Moyenne par blocs de `factor` pas (identique au chargement des données)."""
    x = np.asarray(x, dtype=np.float64)
    if factor <= 1:
        return x
    length = (x.shape[1] // factor) * factor
    return x[:, :length].reshape(x.shape[0], -1, factor).mean(axis=2)


@dataclass
class Member:
    """Un modèle de l'ensemble final et tout ce qu'il faut pour l'appliquer."""

    name: str
    kind: str  # "sklearn" | "tabular_nn" | "hybrid"
    target: TargetTransform
    feature_cols: list[str] = field(default_factory=list)
    preprocessor: TabularPreprocessor | None = None
    model: object | None = None  # MultiTargetModel (kind == "sklearn")
    net_cfg: dict | None = None
    state_dict: dict | None = None
    tab_dim: int = 0
    seq_stats: dict | None = None

    def predict(
        self, feats: pd.DataFrame, temperature: np.ndarray, excitation: np.ndarray
    ) -> np.ndarray:
        """Indicateurs physiques (N, 6)."""
        x_tab = self.preprocessor.transform(feats[self.feature_cols]) if self.preprocessor else None
        if self.kind == "sklearn":
            z = self.model.predict(x_tab)
        else:
            import torch

            from coenergy.nets import (
                build_hybrid_net,
                build_sequence_array,
                build_tabular_net,
                predict_torch,
            )

            if self.kind == "tabular_nn":
                net = build_tabular_net(self.net_cfg, self.tab_dim)
                inputs = (torch.from_numpy(x_tab),)
            else:
                net = build_hybrid_net(self.net_cfg, self.tab_dim)
                seq, _ = build_sequence_array(temperature, excitation, np.arange(1), self.seq_stats)
                inputs = (torch.from_numpy(seq),)
                if self.net_cfg.get("use_tab", True):
                    inputs = inputs + (torch.from_numpy(x_tab),)
            net.load_state_dict(self.state_dict)
            z = predict_torch(net, inputs)
        return self.target.decode(z)


@dataclass
class Predictor:
    """Chaîne de prédiction complète, sérialisable."""

    labels: list[str]
    downsample: int
    dt_minutes: float
    feature_sets: list[str]
    raw_pool: int = 2
    members: list[Member] = field(default_factory=list)
    ensemble: object | None = None  # WeightedEnsemble ou RidgeStacker

    def predict(
        self, temperature: np.ndarray, excitation: np.ndarray, input_dt_minutes: float | None = None
    ) -> pd.DataFrame:
        """Prédit les 6 indicateurs à partir des séries T_int et U.

        Args:
            temperature, excitation: Séries (N, L). Si `input_dt_minutes` est fourni et diffère du
                pas du modèle, elles sont moyennées pour atteindre ce pas.
        """
        # Même précision qu'à l'entraînement : les features sont calculées depuis les tableaux
        # fournis (float32 si les données ont été chargées par `load_simulations`).
        t, u = np.asarray(temperature), np.asarray(excitation)
        if input_dt_minutes is not None:
            factor = int(round(self.dt_minutes / input_dt_minutes))
            t, u = downsample_series(t, factor), downsample_series(u, factor)
        feats = build_features(t, u, self.dt_minutes, self.feature_sets, raw_pool=self.raw_pool)
        preds = {m.name: m.predict(feats, t, u) for m in self.members}
        y = (
            self.ensemble.predict(preds)
            if self.ensemble is not None
            else next(iter(preds.values()))
        )
        return pd.DataFrame(y, columns=self.labels)

    def save(self, path: str | Path) -> None:
        """Sauvegarde dans un fichier joblib unique."""
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> Predictor:
        """Recharge un prédicteur sauvegardé."""
        import joblib

        return joblib.load(path)
