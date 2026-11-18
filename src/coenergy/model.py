"""Architecture CNN-GRU multi-indicateurs.

Inspirée du régresseur CNN-GRU de Nassif et al. (coEnergy) :
- les convolutions 1D captent les motifs locaux (réponse à une marche de chauffe, pentes) et
  réduisent la longueur de la séquence ;
- le GRU capte les dépendances longues dues à l'inertie ;
- trois têtes de sortie, dont une tête Hb positive par construction, dont la somme donne le HLC.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from coenergy.config import ModelConfig


class ConvBlock(nn.Sequential):
    """Conv1d → BatchNorm → GELU → MaxPool (divise la longueur par 2)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(2),
        )


class CNNGRU(nn.Module):
    """Régresseur multi-sorties : séries (T_int, ΔT_int, U) → (Hb, HLC, τ).

    Les sorties sont exprimées dans l'espace normalisé de `TargetScaler` :
    - `hb` : (B, n_hb), positif (softplus) ;
    - `hlc` : (B,), somme exacte de `hb` → contrainte HLC = HLC_ext + HLC_mit garantie ;
    - `tau_z` : (B, n_tau), log des constantes de temps standardisé.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg

        blocks: list[nn.Module] = []
        channels = cfg.in_channels
        for out_channels in cfg.conv_channels:
            blocks.append(ConvBlock(channels, out_channels, cfg.kernel_size))
            channels = out_channels
        self.encoder = nn.Sequential(*blocks)

        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_gru_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_gru_layers > 1 else 0.0,
        )
        self.trunk = nn.Sequential(
            nn.Linear(2 * cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.head_hb = nn.Linear(cfg.hidden_size, cfg.n_hb)
        self.head_tau = nn.Linear(cfg.hidden_size, cfg.n_tau)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Passe avant.

        Args:
            x: Entrées normalisées, forme (B, C, T).

        Returns:
            Dictionnaire avec les clés "hb", "hlc" et "tau_z".
        """
        h = self.encoder(x).transpose(1, 2)  # (B, T', C')
        out, h_n = self.gru(h)
        summary = torch.cat([h_n[-1], out.mean(dim=1)], dim=1)  # état final + moyenne temporelle
        z = self.trunk(summary)
        hb = F.softplus(self.head_hb(z))
        return {"hb": hb, "hlc": hb.sum(dim=1), "tau_z": self.head_tau(z)}
