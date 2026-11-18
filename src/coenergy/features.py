"""Descripteurs physiques des séries temporelles (EDA, baselines et analyse des erreurs).

Ce module n'utilise que NumPy et pandas : il accepte des tableaux NumPy ou des tenseurs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def excitation_descriptors(
    excitation: np.ndarray, dt_minutes: float = 1.0, prbs_threshold: float = 0.9
) -> pd.DataFrame:
    """Résume chaque signal de sollicitation U(t).

    Le type d'excitation est déduit heuristiquement : un PRBS ne prend que deux niveaux
    (0 et P_max), donc la fraction de pas « extrêmes » est proche de 1. Une sinusoïde
    écrêtée passe une partie de son temps entre les deux.

    Args:
        excitation: (N, T) en W.
        dt_minutes: Pas de temps, en minutes.
        prbs_threshold: Fraction minimale de pas extrêmes pour classer en PRBS.

    Returns:
        DataFrame avec p_max (W), energy_kwh, duty_cycle, n_switches, frac_extreme et kind.
    """
    u = np.asarray(excitation, dtype=np.float64)
    p_max = u.max(axis=1)
    tol = np.maximum(1e-6, 0.01 * p_max)[:, None]
    on = u > tol
    extreme = (np.abs(u) <= tol) | (np.abs(u - p_max[:, None]) <= tol)
    frac_extreme = extreme.mean(axis=1)
    return pd.DataFrame(
        {
            "p_max": p_max,
            "energy_kwh": u.sum(axis=1) * dt_minutes / 60.0 / 1000.0,
            "duty_cycle": on.mean(axis=1),
            "n_switches": np.abs(np.diff(on.astype(np.int8), axis=1)).sum(axis=1),
            "frac_extreme": frac_extreme,
            "kind": np.where(frac_extreme >= prbs_threshold, "PRBS", "sinus"),
        }
    )


def temperature_descriptors(temperature: np.ndarray) -> pd.DataFrame:
    """Résume chaque courbe T_int(t).

    Returns:
        DataFrame avec t_initial, t_final, t_max, rise (T_max - T_0), drift (T_fin - T_0), std.
    """
    t = np.asarray(temperature, dtype=np.float64)
    t0 = t[:, 0]
    return pd.DataFrame(
        {
            "t_initial": t0,
            "t_final": t[:, -1],
            "t_max": t.max(axis=1),
            "rise": t.max(axis=1) - t0,
            "drift": t[:, -1] - t0,
            "std": t.std(axis=1),
        }
    )


def summary_features(
    temperature: np.ndarray, excitation: np.ndarray, n_points: int = 24
) -> np.ndarray:
    """Vecteur de descripteurs compact pour une baseline linéaire.

    Concatène la courbe ΔT_int(t) et U(t) moyennées sur `n_points` segments, l'énergie
    injectée et les descripteurs de température.

    Returns:
        Matrice (N, 2 * n_points + 7).
    """
    t = np.asarray(temperature, dtype=np.float64)
    u = np.asarray(excitation, dtype=np.float64)
    n, length = t.shape
    edges = np.linspace(0, length, n_points + 1).astype(int)

    def _segments(x: np.ndarray) -> np.ndarray:
        bounds = zip(edges[:-1], edges[1:], strict=True)
        return np.stack([x[:, a:b].mean(axis=1) for a, b in bounds], axis=1)

    delta = t - t[:, :1]
    temp_desc = temperature_descriptors(t).to_numpy()
    energy = u.sum(axis=1, keepdims=True)
    return np.concatenate([_segments(delta), _segments(u), energy, temp_desc], axis=1)
