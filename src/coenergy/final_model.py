"""Modèle final : entraînement des CNN-GRU multi-résolution et prédicteur d'ensemble livré.

Ce module regroupe le code utilisé pour l'entraînement final sur Kaggle
(`notebooks/modelisation.ipynb`, section 4.7) pour qu'il soit importable, testé et réutilisable :

- `NoisyInput` : augmentation par bruit capteur sur T_int (entraînement du modèle robuste) ;
- `train_sequence_member` : entraîne un CNN-GRU au pas `factor` et renvoie un `Member` ;
- `predict_member` : applique un membre à des séries brutes échantillonnées à la minute ;
- `FinalModel` : charge `modele_final.joblib` / `modele_robuste.joblib` et prédit les 6 indicateurs.

Utilisation :

    from coenergy.final_model import FinalModel
    model = FinalModel.load("results/results_full/modele_final.joblib")
    df = model.predict(T_int, U)      # T_int, U : (N, 480) à la minute → DataFrame (N, 6)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from coenergy.ensemble import WeightedEnsemble
from coenergy.fe import FEATURE_SETS, build_features
from coenergy.nets import (
    TorchSettings,
    build_hybrid_net,
    build_sequence_array,
    fit_torch,
    predict_torch,
)
from coenergy.predict import Member, downsample_series
from coenergy.targets import TargetTransform


class NoisyInput(nn.Module):
    """Ajoute un bruit gaussien de `sigma` °C sur T_int pendant l'entraînement uniquement.

    Les entrées sont standardisées ([T, ΔT, U/1000]) : le même bruit physique `n(t)` est
    appliqué de façon cohérente au canal T (n / std_T) et au canal ΔT ((n - n(0)) / std_ΔT).
    U n'est pas bruité : c'est une commande connue.
    """

    def __init__(self, net: nn.Module, sigma: float, std_t: float, std_dt: float) -> None:
        super().__init__()
        self.net, self.sigma, self.std_t, self.std_dt = net, sigma, std_t, std_dt

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        if self.training and self.sigma > 0:
            x = xs[0]
            n = torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * self.sigma
            noise = torch.cat(
                [n / self.std_t, (n - n[:, :, :1]) / self.std_dt, torch.zeros_like(n)], dim=1
            )
            xs = (x + noise,) + tuple(xs[1:])
        return self.net(*xs)


def train_sequence_member(
    name: str,
    temperature: np.ndarray,
    excitation: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: dict,
    settings: TorchSettings,
    sigma: float = 0.0,
    param: str = "A",
    device: str = "cuda",
) -> tuple[Member, dict]:
    """Entraîne un réseau séquentiel sur des séries déjà ré-échantillonnées au pas du modèle.

    Args:
        name: nom du membre.
        temperature, excitation: séries (N, L) au pas du modèle.
        y: cibles physiques (N, 6) = (HLC, HLC_mit, HLC_ext, τ_b, τ_n, τ_∞).
        train_idx, val_idx: indices d'entraînement et de validation (early stopping).
        cfg: configuration d'architecture (voir `build_hybrid_net`).
        settings: hyperparamètres d'entraînement.
        sigma: écart-type du bruit capteur (°C, au pas du modèle) ; 0 = pas d'augmentation.
        param: paramétrisation des cibles ("A" ou "B", voir `TargetTransform`).
        device: "cuda" ou "cpu".

    Returns:
        Le `Member` prêt à prédire (poids sur CPU) et les informations d'entraînement.
    """
    tt = TargetTransform(param).fit(y[train_idx])
    x, stats = build_sequence_array(temperature, excitation, train_idx)
    xtr = torch.from_numpy(x[train_idx]).to(device)
    xva = torch.from_numpy(x[val_idx]).to(device)
    ztr = torch.from_numpy(tt.encode(y[train_idx])).to(device)
    zva = torch.from_numpy(tt.encode(y[val_idx])).to(device)
    del x
    torch.manual_seed(settings.seed)
    net = build_hybrid_net(cfg, 0).to(device)
    model = NoisyInput(net, sigma, float(stats["std"][0]), float(stats["std"][1])).to(device)
    info = fit_torch(model, (xtr,), ztr, (xva,), zva, settings)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    member = Member(name, "hybrid", tt, [], None, None, cfg, state, 0, stats)
    return member, info


def predict_member(
    member: Member,
    factor: int,
    temperature_1min: np.ndarray,
    excitation_1min: np.ndarray,
    device: str = "cpu",
    batch_size: int = 2048,
) -> np.ndarray:
    """Prédit les 6 indicateurs physiques à partir de séries à la minute (N, 480).

    Les séries sont moyennées par blocs de `factor` minutes, puis standardisées avec les
    statistiques du train du membre (`member.seq_stats`). Si le membre est hybride
    (séquence + variables physiques), les variables sont recalculées au même pas.
    """
    t = downsample_series(temperature_1min, factor).astype(np.float32)
    u = downsample_series(excitation_1min, factor).astype(np.float32)
    net = build_hybrid_net(member.net_cfg, member.tab_dim)
    net.load_state_dict(member.state_dict)
    net.to(device)
    seq, _ = build_sequence_array(t, u, np.arange(1), member.seq_stats)
    inputs: tuple[torch.Tensor, ...] = (torch.from_numpy(seq).to(device),)
    if member.preprocessor is not None and member.net_cfg.get("use_tab", False):
        feats = build_features(t, u, float(factor), sets=FEATURE_SETS, raw_pool=2)
        x_tab = member.preprocessor.transform(feats[member.feature_cols])
        inputs = inputs + (torch.from_numpy(x_tab).to(device),)
    z = predict_torch(net, inputs, batch_size=batch_size)
    return member.target.decode(z)


@dataclass
class FinalModel:
    """Ensemble livré : membres (pas en minutes, Member) et combinaison pondérée par composante.

    La contrainte physique HLC = Hb_mit + Hb_ext est garantie : chaque membre prédit les deux Hb
    et l'ensemble combine Hb_mit et Hb_ext séparément avant de recalculer HLC.
    """

    labels: list[str]
    names: list[str]
    members: list[tuple[int, Member]]
    combiner: WeightedEnsemble | None = None

    @staticmethod
    def load(path: str | Path) -> FinalModel:
        """Charge un bundle sauvegardé par le notebook d'entraînement final."""
        b = joblib.load(path)
        return FinalModel(list(b["labels"]), list(b["names"]), list(b["members"]), b["combiner"])

    def predict_array(
        self, temperature_1min: np.ndarray, excitation_1min: np.ndarray, device: str = "cpu"
    ) -> np.ndarray:
        """Prédictions (N, 6) à partir de T_int et U échantillonnés à la minute."""
        preds = {
            name: predict_member(m, factor, temperature_1min, excitation_1min, device)
            for name, (factor, m) in zip(self.names, self.members, strict=True)
        }
        if self.combiner is None:
            return preds[self.names[0]]
        return self.combiner.predict(preds)

    def predict(
        self, temperature_1min: np.ndarray, excitation_1min: np.ndarray, device: str = "cpu"
    ) -> pd.DataFrame:
        """Prédictions sous forme de DataFrame (colonnes = indicateurs)."""
        return pd.DataFrame(
            self.predict_array(temperature_1min, excitation_1min, device), columns=self.labels
        )
