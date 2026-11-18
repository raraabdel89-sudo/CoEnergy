"""Feature engineering pour les modèles tabulaires (GBM, MLP, forêts…).

Trois familles de descripteurs, toutes calculées simulation par simulation (aucune information
d'une simulation ne se mélange à une autre, donc aucun risque de fuite entre train et test) :

- `physics_features` : descripteurs physiques de l'excitation et de la réponse (énergie,
  commutations, montée, pentes, réponse après le premier allumage, dérive libre avant chauffe…) ;
- `arx_features` : identification d'un modèle ARX(n, n) par moindres carrés sur chaque
  simulation, dont on déduit un gain statique (≈ 1/HLC), des pôles et des constantes de temps.
  C'est la transposition « identification de système » de la physique du problème ;
- `raw_features` : les séries ΔT_int(t) et U(t) elles-mêmes, éventuellement moyennées.

`TabularPreprocessor` gère ensuite l'imputation, l'écrêtage et la normalisation, ajustés sur
le seul jeu d'entraînement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _as_f64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------------------
# Descripteurs physiques
# --------------------------------------------------------------------------------------
def physics_features(
    temperature: np.ndarray,
    excitation: np.ndarray,
    dt_minutes: float,
    n_bins: int = 8,
    onset_lags_min: Sequence[float] = (15, 30, 60, 120),
) -> pd.DataFrame:
    """Descripteurs physiques de l'excitation et de la réponse thermique.

    Args:
        temperature: T_int, forme (N, L), en °C.
        excitation: U, forme (N, L), en W.
        dt_minutes: Pas de temps, en minutes.
        n_bins: Nombre de segments temporels pour les moyennes de ΔT.
        onset_lags_min: Délais (min) auxquels on mesure la réponse après le premier allumage.

    Returns:
        DataFrame (N, ~35). Les valeurs non définies (ex. pas de chauffe) valent NaN.
    """
    t = _as_f64(temperature)
    u = _as_f64(excitation)
    n, length = t.shape
    dt_h = dt_minutes / 60.0
    time_h = np.arange(length) * dt_h
    d = t - t[:, :1]
    on = u > 1.0
    has_heat = on.any(axis=1)
    energy_kwh = u.sum(axis=1) * dt_h / 1000.0

    feats: dict[str, np.ndarray] = {
        "energy_kwh": energy_kwh,
        "duty_cycle": on.mean(axis=1),
        "n_switches": np.abs(np.diff(on.astype(np.int8), axis=1)).sum(axis=1).astype(float),
        "rise": d.max(axis=1),
        "drift": d[:, -1],
        "t_std": t.std(axis=1),
        "integral_dT_Kh": d.sum(axis=1) * dt_h,
    }

    first_on = np.where(has_heat, on.argmax(axis=1), length)
    feats["first_on_h"] = first_on * dt_h
    last_switch = np.full(n, np.nan)
    sw = np.abs(np.diff(on.astype(np.int8), axis=1)) > 0
    any_sw = sw.any(axis=1)
    last_switch[any_sw] = (length - 1 - np.argmax(sw[:, ::-1], axis=1)[any_sw]) * dt_h
    feats["time_since_last_switch_h"] = np.where(any_sw, time_h[-1] - last_switch, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        u_sum = u.sum(axis=1)
        feats["heat_weighted_dT"] = np.where(u_sum > 0, (d * u).sum(axis=1) / u_sum, np.nan)
        feats["heat_time_centroid_h"] = np.where(
            u_sum > 0, (u * time_h).sum(axis=1) / u_sum, np.nan
        )
        feats["drift_per_kwh"] = np.where(energy_kwh > 0, d[:, -1] / energy_kwh, np.nan)
        feats["rise_per_kwh"] = np.where(energy_kwh > 0, d.max(axis=1) / energy_kwh, np.nan)

        # Vitesse moyenne de variation pendant la chauffe / à l'arrêt (K/h)
        dd = np.diff(t, axis=1) / dt_h
        on_mid = on[:, :-1]
        n_on, n_off = on_mid.sum(axis=1), (~on_mid).sum(axis=1)
        feats["mean_rate_on"] = np.where(
            n_on > 0, (dd * on_mid).sum(axis=1) / np.maximum(n_on, 1), np.nan
        )
        feats["mean_rate_off"] = np.where(
            n_off > 0, (dd * ~on_mid).sum(axis=1) / np.maximum(n_off, 1), np.nan
        )

    # Pente sur la dernière heure (régression linéaire)
    n_last = max(2, int(round(60 / dt_minutes)))
    tt = time_h[-n_last:] - time_h[-n_last:].mean()
    feats["slope_last_hour"] = (t[:, -n_last:] * tt).sum(axis=1) / (tt**2).sum()

    # Évolution libre avant le premier allumage : renseigne sur l'état initial des parois
    pre_slope = np.full(n, np.nan)
    for k in np.unique(first_on):
        if k < 3:
            continue
        rows = first_on == k
        tk = time_h[:k] - time_h[:k].mean()
        pre_slope[rows] = (t[rows, :k] * tk).sum(axis=1) / (tk**2).sum()
    feats["pre_heat_slope"] = pre_slope
    feats["pre_heat_duration_h"] = np.where(has_heat, first_on * dt_h, np.nan)

    # Réponse après le premier allumage
    idx0 = np.minimum(first_on, length - 1)
    t0 = t[np.arange(n), idx0]
    for lag in onset_lags_min:
        k = first_on + int(round(lag / dt_minutes))
        valid = has_heat & (k < length)
        val = np.full(n, np.nan)
        val[valid] = t[np.where(valid)[0], k[valid]] - t0[valid]
        feats[f"onset_resp_{int(lag)}min"] = val

    # Moyennes de ΔT par segment temporel
    edges = np.linspace(0, length, n_bins + 1).astype(int)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        feats[f"dT_bin{i}"] = d[:, a:b].mean(axis=1)

    return pd.DataFrame(feats)


# --------------------------------------------------------------------------------------
# Identification ARX
# --------------------------------------------------------------------------------------
def arx_features(
    temperature: np.ndarray,
    excitation: np.ndarray,
    dt_minutes: float,
    order: int = 2,
    ridge: float = 1e-6,
    chunk: int = 10_000,
) -> pd.DataFrame:
    """Ajuste, sur chaque simulation, un modèle ARX(n, n) par moindres carrés :

        ΔT_k = Σ_i a_i ΔT_{k-i} + Σ_j b_j u_{k-j} + c      (u en kW)

    et en déduit des grandeurs physiques :
    - gain statique G = Σb / (1 - Σa) en K/kW, et H_arx = 1000 / G en W/K (analogue du HLC) ;
    - température d'équilibre libre ΔT_eq = c / (1 - Σa) (dérive liée à l'état initial) ;
    - pôles du système (valeurs propres de la matrice compagnon) et constantes de temps
      τ_i = -Δt / ln|p_i| en heures ;
    - qualité d'ajustement (RMSE résiduel, R²).

    Les simulations sans chauffe n'identifient pas b : leurs grandeurs liées au gain valent NaN.

    Returns:
        DataFrame (N, 2n + 8) préfixé par `arx{n}_`.
    """
    t = _as_f64(temperature)
    u = _as_f64(excitation) / 1000.0
    n_sim, length = t.shape
    p = 2 * order + 1
    dt_h = dt_minutes / 60.0
    out = {
        k: np.full(n_sim, np.nan)
        for k in ["gain_K_per_kW", "H_W_per_K", "dT_eq", "rmse", "r2", "max_pole", "complex_poles"]
    }
    coefs = np.full((n_sim, 2 * order), np.nan)
    taus = np.full((n_sim, order), np.nan)

    for s in range(0, n_sim, chunk):
        y = t[s : s + chunk] - t[s : s + chunk, :1]
        uu = u[s : s + chunk]
        m = y.shape[0]
        cols = [y[:, order - i : length - i] for i in range(1, order + 1)]
        cols += [uu[:, order - j : length - j] for j in range(1, order + 1)]
        cols.append(np.ones_like(cols[0]))
        x = np.stack(cols, axis=2)  # (m, K, p)
        target = y[:, order:]
        xtx = np.einsum("nkp,nkq->npq", x, x)
        xty = np.einsum("nkp,nk->np", x, target)
        scale = np.trace(xtx, axis1=1, axis2=2) / p
        reg = ridge * scale[:, None, None] * np.eye(p)[None]
        theta = np.linalg.solve(xtx + reg, xty[..., None])[..., 0]  # (m, p)

        pred = np.einsum("nkp,np->nk", x, theta)
        resid = target - pred
        var = target.var(axis=1)
        out["rmse"][s : s + m] = np.sqrt((resid**2).mean(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            out["r2"][s : s + m] = np.where(var > 1e-12, 1 - (resid**2).mean(axis=1) / var, np.nan)

        a, b, c = theta[:, :order], theta[:, order : 2 * order], theta[:, -1]
        coefs[s : s + m] = theta[:, : 2 * order]
        sa, sb = a.sum(axis=1), b.sum(axis=1)
        heated = uu.max(axis=1) > 1e-3
        with np.errstate(invalid="ignore", divide="ignore"):
            gain = sb / (1 - sa)
            valid = heated & (gain > 1e-6) & (sa < 1)
            out["gain_K_per_kW"][s : s + m] = np.where(valid, gain, np.nan)
            out["H_W_per_K"][s : s + m] = np.where(valid, 1000.0 / gain, np.nan)
            out["dT_eq"][s : s + m] = np.where(sa < 1, c / (1 - sa), np.nan)

        # Pôles : valeurs propres de la matrice compagnon
        comp = np.zeros((m, order, order))
        comp[:, 0, :] = a
        if order > 1:
            comp[:, np.arange(1, order), np.arange(order - 1)] = 1.0
        poles = np.linalg.eigvals(comp)
        mod = np.abs(poles)
        out["max_pole"][s : s + m] = mod.max(axis=1)
        out["complex_poles"][s : s + m] = (np.abs(poles.imag) > 1e-9).any(axis=1).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            tau = np.where((mod > 1e-9) & (mod < 1), -dt_h / np.log(mod), np.nan)
        taus[s : s + m] = -np.sort(-np.nan_to_num(tau, nan=-np.inf), axis=1)
    taus[~np.isfinite(taus)] = np.nan

    df = pd.DataFrame(out)
    for i in range(order):
        df[f"a{i + 1}"] = coefs[:, i]
        df[f"b{i + 1}"] = coefs[:, order + i]
        df[f"tau{i + 1}_h"] = taus[:, i]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_tau1"] = np.log(df["tau1_h"])
        df["log_H"] = np.log(df["H_W_per_K"])
    return df.add_prefix(f"arx{order}_")


# --------------------------------------------------------------------------------------
# Séries brutes
# --------------------------------------------------------------------------------------
def raw_features(temperature: np.ndarray, excitation: np.ndarray, pool: int = 1) -> pd.DataFrame:
    """Séries ΔT_int(t) et U(t) (en kW), moyennées par blocs de `pool` pas."""
    t = _as_f64(temperature)
    u = _as_f64(excitation) / 1000.0
    d = t - t[:, :1]
    if pool > 1:
        length = (d.shape[1] // pool) * pool
        d = d[:, :length].reshape(d.shape[0], -1, pool).mean(axis=2)
        u = u[:, :length].reshape(u.shape[0], -1, pool).mean(axis=2)
    cols = {f"dT_{i:03d}": d[:, i] for i in range(d.shape[1])}
    cols.update({f"U_{i:03d}": u[:, i] for i in range(u.shape[1])})
    return pd.DataFrame(cols)


FEATURE_SETS = ("phys", "arx1", "arx2", "arx3", "raw")


def build_features(
    temperature: np.ndarray,
    excitation: np.ndarray,
    dt_minutes: float,
    sets: Sequence[str] = ("phys", "arx2"),
    raw_pool: int = 2,
    chunk: int = 50_000,
) -> pd.DataFrame:
    """Assemble les familles de descripteurs demandées, par morceaux pour limiter la mémoire.

    Args:
        sets: Sous-ensemble de {"phys", "arx1", "arx2", "arx3", "raw"}.
        raw_pool: Facteur de moyennage des séries brutes.
    """
    unknown = set(sets) - set(FEATURE_SETS)
    if unknown:
        raise ValueError(f"Familles inconnues : {unknown}")
    parts = []
    for s in range(0, temperature.shape[0], chunk):
        tt, uu = temperature[s : s + chunk], excitation[s : s + chunk]
        block = []
        if "phys" in sets:
            block.append(physics_features(tt, uu, dt_minutes))
        for k in (1, 2, 3):
            if f"arx{k}" in sets:
                block.append(arx_features(tt, uu, dt_minutes, order=k))
        if "raw" in sets:
            block.append(raw_features(tt, uu, pool=raw_pool))
        parts.append(pd.concat(block, axis=1))
    return pd.concat(parts, ignore_index=True).astype(np.float32)


def select_columns(df: pd.DataFrame, families: Sequence[str]) -> list[str]:
    """Colonnes d'un DataFrame de features appartenant aux familles demandées."""
    cols = []
    for c in df.columns:
        fam = (
            "raw"
            if c.startswith(("dT_0", "dT_1", "U_"))
            else (c.split("_")[0] if c.startswith("arx") else "phys")
        )
        if fam in families:
            cols.append(c)
    return cols


# --------------------------------------------------------------------------------------
# Prétraitement tabulaire
# --------------------------------------------------------------------------------------
@dataclass
class TabularPreprocessor:
    """Imputation (médiane + indicateurs de valeur manquante), écrêtage et normalisation.

    Toutes les statistiques sont ajustées sur le seul jeu d'entraînement.

    Attributes:
        scaling: "standard", "robust" (médiane / IQR) ou "quantile" (gaussianisation par rangs).
        clip_quantile: Écrêtage aux quantiles q et 1-q (0 pour désactiver).
    """

    scaling: str = "standard"
    clip_quantile: float = 0.001
    columns: list[str] = field(default_factory=list)
    medians: np.ndarray | None = None
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    center: np.ndarray | None = None
    scale: np.ndarray | None = None
    nan_cols: list[int] = field(default_factory=list)
    quantile_grid: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> TabularPreprocessor:
        """Ajuste les statistiques sur le train."""
        x = df.to_numpy(np.float64)
        self.columns = list(df.columns)
        self.medians = np.nanmedian(x, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        self.nan_cols = [j for j in range(x.shape[1]) if np.isnan(x[:, j]).any()]
        x = self._impute(x)
        if self.clip_quantile > 0:
            self.lo = np.quantile(x, self.clip_quantile, axis=0)
            self.hi = np.quantile(x, 1 - self.clip_quantile, axis=0)
            x = np.clip(x, self.lo, self.hi)
        if self.scaling == "standard":
            self.center, self.scale = x.mean(axis=0), x.std(axis=0)
        elif self.scaling == "robust":
            q1, q3 = np.quantile(x, [0.25, 0.75], axis=0)
            self.center, self.scale = np.median(x, axis=0), q3 - q1
        elif self.scaling == "quantile":
            self.quantile_grid = np.quantile(x, np.linspace(0, 1, 1001), axis=0)
        else:
            raise ValueError(self.scaling)
        if self.scale is not None:
            self.scale = np.where(self.scale > 1e-12, self.scale, 1.0)
        return self

    def _impute(self, x: np.ndarray) -> np.ndarray:
        assert self.medians is not None
        x = np.where(np.isnan(x), self.medians[None, :], x)
        return np.where(np.isfinite(x), x, self.medians[None, :])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Renvoie une matrice float32 prête pour un modèle."""
        raw = df[self.columns].to_numpy(np.float64)
        indicators = np.isnan(raw[:, self.nan_cols]).astype(np.float64)
        x = self._impute(raw)
        if self.lo is not None:
            x = np.clip(x, self.lo, self.hi)
        if self.scaling == "quantile":
            from scipy.special import ndtri

            grid = self.quantile_grid
            assert grid is not None
            levels = np.linspace(0, 1, grid.shape[0])
            z = np.empty_like(x)
            for j in range(x.shape[1]):
                g = grid[:, j]
                if g[-1] - g[0] < 1e-12:
                    z[:, j] = 0.0
                    continue
                q = np.interp(x[:, j], g, levels)
                z[:, j] = ndtri(np.clip(q, 1e-4, 1 - 1e-4))
            x = z
        else:
            x = (x - self.center) / self.scale
        return np.concatenate([x, indicators], axis=1).astype(np.float32)

    @property
    def output_names(self) -> list[str]:
        """Noms des colonnes produites par `transform`."""
        return self.columns + [f"isnan_{self.columns[j]}" for j in self.nan_cols]
