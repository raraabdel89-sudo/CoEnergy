"""Fonctions utilitaires : reproductibilité et choix du device."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int = 42) -> None:
    """Fixe toutes les graines aléatoires (Python, NumPy, PyTorch).

    Args:
        seed: Valeur de la graine.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Renvoie le device de calcul.

    Args:
        preference: "auto", "cpu", "cuda" ou "mps".

    Returns:
        Le device PyTorch à utiliser.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Compte les paramètres entraînables d'un modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
