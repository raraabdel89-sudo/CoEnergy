"""Paramétrisations des cibles pour les modèles tabulaires et hybrides.

Les modèles prédisent un vecteur z de 5 composantes standardisées ; `decode` reconstruit les
6 indicateurs physiques (HLC, Hb[0], Hb[1], τ_b, τ_n, τ_∞) en garantissant par construction :
HLC = Hb[0] + Hb[1], positivité de toutes les grandeurs.

- Paramétrisation "A" (directe) : z ← [Hb0/s, Hb1/s, log τ_b, log τ_n, log τ_∞].
- Paramétrisation "B" (physique, issue de l'EDA) : z ← [log HLC, part Hb0/HLC,
  log(τ_b·HLC), log(τ_n·HLC), log(τ_∞·HLC)]. Comme τ ≈ C/H avec C indépendante de H
  (section 3.12), τ·HLC est une capacité effective : les composantes sont quasi décorrélées.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z_NAMES = {
    "A": ["hb0_scaled", "hb1_scaled", "log_tau_b", "log_tau_n", "log_tau_inf"],
    "B": ["log_HLC", "share_hb0", "log_Ceff_b", "log_Ceff_n", "log_Ceff_inf"],
}


@dataclass
class TargetTransform:
    """Encodage / décodage des 6 indicateurs (ordre : HLC, Hb0, Hb1, τ_b, τ_n, τ_∞)."""

    param: str = "A"
    hb_scale: float = 1.0
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def _raw(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        hlc, hb0, hb1, tau = y[:, 0], y[:, 1], y[:, 2], np.clip(y[:, 3:6], 1e-6, None)
        if self.param == "A":
            return np.column_stack([hb0 / self.hb_scale, hb1 / self.hb_scale, np.log(tau)])
        if self.param == "B":
            return np.column_stack([np.log(hlc), hb0 / hlc, np.log(tau * hlc[:, None])])
        raise ValueError(self.param)

    def fit(self, y_train: np.ndarray) -> TargetTransform:
        """Ajuste l'échelle et la standardisation sur le train."""
        self.hb_scale = float(np.mean(y_train[:, 0]))
        raw = self._raw(y_train)
        self.mean, self.std = raw.mean(axis=0), raw.std(axis=0) + 1e-12
        return self

    def encode(self, y: np.ndarray) -> np.ndarray:
        """Indicateurs physiques (N, 6) → z standardisé (N, 5), float32."""
        return ((self._raw(y) - self.mean) / self.std).astype(np.float32)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """z (N, 5) → indicateurs physiques (N, 6), cohérents par construction."""
        raw = np.asarray(z, dtype=np.float64) * self.std + self.mean
        if self.param == "A":
            hb0 = np.clip(raw[:, 0] * self.hb_scale, 1e-3, None)
            hb1 = np.clip(raw[:, 1] * self.hb_scale, 1e-3, None)
            hlc = hb0 + hb1
            tau = np.exp(raw[:, 2:5])
        else:
            hlc = np.exp(raw[:, 0])
            share = np.clip(raw[:, 1], 1e-3, 1 - 1e-3)
            hb0, hb1 = share * hlc, (1 - share) * hlc
            tau = np.exp(raw[:, 2:5]) / hlc[:, None]
        return np.column_stack([hlc, hb0, hb1, tau])

    @property
    def names(self) -> list[str]:
        """Noms des composantes de z."""
        return Z_NAMES[self.param]
