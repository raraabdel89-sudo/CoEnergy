"""Fonction de perte multi-tâches."""

from __future__ import annotations

import torch

from coenergy.data import Batch, TargetScaler


def multitask_loss(
    pred: dict[str, torch.Tensor],
    batch: Batch,
    target_scaler: TargetScaler,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Somme pondérée de MSE normalisées par famille d'indicateurs.

    Chaque erreur est divisée par l'écart-type de sa cible (calculé sur le train), de sorte
    que les trois familles (Hb, HLC, τ) contribuent à des échelles comparables.

    Args:
        pred: Sorties du modèle.
        batch: Batch contenant les cibles normalisées.
        target_scaler: Fournit les écarts-types de normalisation.
        weights: Poids des termes "hb", "hlc" et "tau".

    Returns:
        La perte totale et le détail de chaque terme.
    """
    hb_std = target_scaler.hb_std.to(pred["hb"].device)
    parts = {
        "hb": (((pred["hb"] - batch.hb) / hb_std) ** 2).mean(),
        "hlc": (((pred["hlc"] - batch.hlc) / target_scaler.hlc_std) ** 2).mean(),
        "tau": ((pred["tau_z"] - batch.tau_z) ** 2).mean(),
    }
    total = sum(weights.get(k, 1.0) * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
