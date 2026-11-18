"""Variantes testées dans l'annexe A3 : réponse moyenne de la population et GRU bidirectionnel.

Version réutilisable et testée du code écrit dans `notebooks/annexes/A3_pistes_amelioration.ipynb`
pendant son exécution sur Kaggle :

- `fit_population_response` / `population_prediction` : modèle linéaire commun à tous les
  bâtiments, E[ΔT(t) | U] = Σ_k h[k]·u(t−k) + f[t], dont le résidu isole la signature propre
  à chaque bâtiment ;
- `standardize_channels` : empilement et standardisation de canaux avec les statistiques du train ;
- `BiGRUEncoder` : convolutions 1D puis GRU bidirectionnel (encodeur compatible avec `HybridNet`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy.signal import lfilter
from torch import nn


def fit_population_response(
    delta_t: np.ndarray,
    u_kw: np.ndarray,
    idx: np.ndarray,
    n_sub: int = 60_000,
    ridge: float = 1e-4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Ajuste E[ΔT(t) | U] = Σ_k h[k]·u(t−k) + f[t] par moindres carrés régularisés.

    Args:
        delta_t: montée de température ΔT(t) = T(t) − T(0), forme (N, L).
        u_kw: puissance de chauffe en kW, forme (N, L).
        idx: indices utilisables pour l'ajustement (le train).
        n_sub: nombre de simulations tirées dans `idx` (limite la mémoire et le temps).
        ridge: régularisation relative (fraction de la trace moyenne de XᵀX).
        seed: graine du tirage.

    Returns:
        `h` (L,) : réponse impulsionnelle moyenne ; `f` (L,) : dérive moyenne sans chauffage.
    """
    sub = np.random.default_rng(seed).choice(idx, size=min(n_sub, len(idx)), replace=False)
    y, u = delta_t[sub], u_kw[sub]
    length = y.shape[1]
    xtx, xty = np.zeros((2 * length, 2 * length)), np.zeros(2 * length)
    for t in range(length):
        a = np.zeros((len(sub), length))
        a[:, : t + 1] = u[:, t::-1]  # u(t), u(t−1), …, u(0)
        s = a.sum(axis=0)
        xtx[:length, :length] += a.T @ a
        xtx[:length, length + t] += s
        xtx[length + t, :length] += s
        xtx[length + t, length + t] += len(sub)
        xty[:length] += a.T @ y[:, t]
        xty[length + t] += y[:, t].sum()
    reg = ridge * np.trace(xtx) / (2 * length) * np.eye(2 * length)
    theta = np.linalg.solve(xtx + reg, xty)
    return theta[:length], theta[length:]


def population_prediction(u_kw: np.ndarray, h: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Réponse moyenne prédite pour chaque excitation : convolution par `h` plus la dérive `f`."""
    return lfilter(h, [1.0], u_kw, axis=1) + f[None, :]


def standardize_channels(channels: Sequence[np.ndarray], idx: np.ndarray) -> np.ndarray:
    """Empile des canaux (N, L) en (N, C, L) et les standardise avec les statistiques de `idx`."""
    x = np.stack(channels, axis=1).astype(np.float32)
    m, s = x[idx].mean(axis=(0, 2)), x[idx].std(axis=(0, 2)) + 1e-6
    return (x - m[None, :, None]) / s[None, :, None]


class BiGRUEncoder(nn.Module):
    """Convolutions 1D puis GRU bidirectionnel (lecture de la série dans les deux sens).

    La représentation concatène les états finaux des deux sens et la moyenne temporelle des
    sorties : dimension `out_features = 4 × hidden`.
    """

    def __init__(
        self,
        in_ch: int,
        channels: int = 128,
        n_conv: int = 2,
        kernel: int = 3,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        c = in_ch
        for _ in range(n_conv):
            blocks += [
                nn.Conv1d(c, channels, kernel, padding=kernel // 2),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.MaxPool1d(2),
            ]
            c = channels
        self.conv = nn.Sequential(*blocks)
        self.gru = nn.GRU(
            channels,
            hidden,
            layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out_features = 4 * hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(self.conv(x).transpose(1, 2))
        return torch.cat([h_n[-2], h_n[-1], out.mean(dim=1)], dim=1)
