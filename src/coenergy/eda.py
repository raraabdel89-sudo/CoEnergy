"""Analyses exploratoires avancées : plan d'expérience, doublons, identifiabilité, spectre.

Uniquement NumPy et pandas (pas de SciPy ni scikit-learn) : les corrélations de Spearman sont
calculées comme des corrélations de Pearson sur les rangs moyens.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- corrélations
def _ranks(x: np.ndarray) -> np.ndarray:
    """Rangs moyens (gestion des ex aequo) colonne par colonne."""
    return pd.DataFrame(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy()


def spearman_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Corrélations de Spearman entre chaque colonne de `a` (N, p) et de `b` (N, q) -> (p, q).

    Une colonne constante donne NaN.
    """
    ra, rb = _ranks(np.atleast_2d(a.T).T), _ranks(np.atleast_2d(b.T).T)
    ra = ra - ra.mean(0)
    rb = rb - rb.mean(0)
    num = ra.T @ rb
    den = np.sqrt((ra**2).sum(0))[:, None] * np.sqrt((rb**2).sum(0))[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def spearman_table(
    features: pd.DataFrame, y: np.ndarray, labels: Sequence[str]
) -> pd.DataFrame:
    """Tableau de Spearman descripteurs x indicateurs."""
    corr = spearman_matrix(features.to_numpy(dtype=np.float64), y)
    return pd.DataFrame(corr, index=features.columns, columns=list(labels))


# --------------------------------------------------------------------------- homogénéité
def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Statistique de Kolmogorov-Smirnov à deux échantillons : max |F_a - F_b|."""
    a, b = np.sort(np.asarray(a, dtype=np.float64)), np.sort(np.asarray(b, dtype=np.float64))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def ks_critical(n_a: int, n_b: int, alpha: float = 0.05) -> float:
    """Seuil asymptotique du test KS à deux échantillons (rejet si statistique > seuil)."""
    c = np.sqrt(-0.5 * np.log(alpha / 2))
    return float(c * np.sqrt((n_a + n_b) / (n_a * n_b)))


def distribution_shift(
    y: np.ndarray, groups: np.ndarray, labels: Sequence[str], reference: object | None = None
) -> pd.DataFrame:
    """Statistique KS de chaque groupe contre la référence (le reste des données par défaut).

    Returns:
        Une ligne par groupe, une colonne par indicateur, plus le seuil KS à 5 %.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    rows, thresholds = {}, {}
    for g in np.unique(groups):
        in_g = groups == g
        ref = (groups == reference) if reference is not None else ~in_g
        if reference is not None and g == reference:
            continue
        rows[g] = [ks_statistic(y[in_g, j], y[ref, j]) for j in range(y.shape[1])]
        thresholds[g] = ks_critical(int(in_g.sum()), int(ref.sum()))
    out = pd.DataFrame(rows, index=list(labels)).T
    out["seuil KS 5 %"] = pd.Series(thresholds)
    return out


# --------------------------------------------------------------------------- plan d'expérience
def design_summary(solicitation: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Résumé des paramètres de génération de la sollicitation (un par simulation)."""
    df = pd.DataFrame(np.asarray(solicitation, dtype=np.float64), columns=list(names))
    out = pd.DataFrame(
        {
            "n_valeurs_distinctes": df.nunique(),
            "min": df.min(),
            "médiane": df.median(),
            "max": df.max(),
        }
    )
    out["statut"] = np.where(out["n_valeurs_distinctes"] == 1, "constant", "aléatoire")
    return out


def excitation_groups(excitation: np.ndarray) -> np.ndarray:
    """Identifiant de groupe pour chaque simulation : même identifiant = même signal U(t)."""
    u = np.ascontiguousarray(np.asarray(excitation, dtype=np.float32))
    keys = u.view(np.dtype((np.void, u.dtype.itemsize * u.shape[1]))).ravel()
    _, inverse = np.unique(keys, return_inverse=True)
    return inverse.ravel()


def duplicate_report(t_keys: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.Series:
    """Doublons exacts sur les courbes T_int, les cibles et les signaux U(t).

    Args:
        t_keys: Courbes T_int (N, T) ou leurs empreintes (N,), voir `scan.row_hashes`.
        y: Indicateurs (N, k).
        groups: Identifiant du motif U(t) de chaque simulation.
    """
    keys = np.asarray(t_keys)
    t_dup = (pd.DataFrame(keys) if keys.ndim == 2 else pd.Series(keys)).duplicated().sum()
    sizes = np.bincount(groups)
    return pd.Series(
        {
            "simulations": len(y),
            "motifs U(t) distincts": int((sizes > 0).sum()),
            "part des simulations partageant leur U(t)": float((sizes[groups] > 1).mean()),
            "taille du plus grand groupe U(t)": int(sizes.max()),
            "courbes T_int(t) dupliquées": int(t_dup),
            "vecteurs d'indicateurs dupliqués": int(pd.DataFrame(np.asarray(y)).duplicated().sum()),
        }
    )


def group_split_overlap(groups: np.ndarray, splits: dict[str, np.ndarray]) -> pd.Series:
    """Part des simulations de test dont le motif U(t) apparaît aussi dans le train."""
    train_groups = np.unique(groups[np.asarray(splits["train"])])
    out = {}
    for name in ("val", "test"):
        g = groups[np.asarray(splits[name])]
        out[f"{name} : motif U déjà vu au train"] = float(np.isin(g, train_groups).mean())
    return pd.Series(out)


# --------------------------------------------------------------------------- richesse du signal
def excitation_richness(excitation: np.ndarray, dt_minutes: float = 1.0) -> pd.DataFrame:
    """Descripteurs d'informativité de U(t) : chauffe nulle, instant d'allumage, nb de fronts."""
    u = np.asarray(excitation)
    on = u > 0
    any_on = on.any(axis=1)
    first_on = np.where(any_on, on.argmax(axis=1), u.shape[1]) * dt_minutes / 60
    switches = np.abs(np.diff(on.astype(np.int8), axis=1)).sum(axis=1)
    category = np.select(
        [~any_on, switches <= 1, switches <= 4],
        ["aucune chauffe", "un seul front (échelon)", "2 à 4 fronts"],
        "5 fronts et plus",
    )
    return pd.DataFrame(
        {"first_on_h": first_on, "n_switches": switches, "duty_cycle": on.mean(axis=1),
         "categorie": category}
    )


def mean_power_spectrum(
    x: np.ndarray, dt_minutes: float = 1.0, detrend: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Densité spectrale moyenne (périodogramme avec fenêtre de Hann).

    Returns:
        Fréquences en cycles/heure (sans la composante continue) et DSP moyenne normalisée
        (somme = 1).
    """
    s = np.asarray(x, dtype=np.float64)
    if detrend:
        s = s - s.mean(axis=1, keepdims=True)
    window = np.hanning(s.shape[1])
    psd = np.abs(np.fft.rfft(s * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(s.shape[1], d=dt_minutes / 60)
    mean_psd = psd.mean(axis=0)[1:]
    return freqs[1:], mean_psd / mean_psd.sum()


# --------------------------------------------------------------------------- identification RC
def fit_rc_model(
    temperature: np.ndarray, excitation: np.ndarray, dt_minutes: float = 1.0
) -> pd.DataFrame:
    """Identifie un modèle « boîte grise » R1C1 par moindres carrés, simulation par simulation.

    Discrétisation d'Euler de C dT/dt = U - H (T - T_ref) :
        T[k+1] - T[k] = θ0 + θ1 T[k] + θ2 U[k]
    d'où H = -θ1/θ2 (W/K), C = dt/θ2 et τ = C/H = -dt/θ1.

    Les simulations sans chauffe ne permettent pas d'estimer θ2 : H et C valent NaN.

    Returns:
        DataFrame avec H_rc (W/K), tau_rc (h), C_rc (kWh/K) et r2 (qualité d'ajustement de ΔT).
    """
    t = np.asarray(temperature, dtype=np.float64)
    u = np.asarray(excitation, dtype=np.float64) / 1000.0  # kW : meilleur conditionnement
    dt_h = dt_minutes / 60
    target = np.diff(t, axis=1)
    design = np.stack([np.ones_like(t[:, :-1]), t[:, :-1], u[:, :-1]], axis=2)
    xtx = np.einsum("nki,nkj->nij", design, design)
    xty = np.einsum("nki,nk->ni", design, target)
    heated = u.max(axis=1) > 0
    theta = np.full((len(t), 3), np.nan)
    ridge = 1e-9 * np.eye(3)
    theta[heated] = np.linalg.solve(xtx[heated] + ridge, xty[heated][..., None])[..., 0]
    resid = target - np.einsum("nki,ni->nk", design, np.nan_to_num(theta))
    ss_tot = ((target - target.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    # Un ajustement n'est physique que si θ1 < 0 (relaxation) et θ2 > 0 (la chauffe réchauffe)
    physical = heated & (theta[:, 1] < 0) & (theta[:, 2] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(physical, -theta[:, 1] / theta[:, 2] * 1000.0, np.nan)
        tau = np.where(physical, -dt_h / theta[:, 1], np.nan)
        cap = np.where(physical, dt_h / theta[:, 2], np.nan)
        r2 = np.where(heated, 1 - (resid**2).sum(axis=1) / ss_tot, np.nan)
    return pd.DataFrame({"H_rc": h, "tau_rc": tau, "C_rc": cap, "r2": r2, "physique": physical})


def variance_share_by_group(
    temperature: np.ndarray, groups: np.ndarray, step: int = 10, dt_minutes: float = 1.0,
    min_size: int = 2
) -> pd.Series:
    """η²(t) : part de la variance de T_int(t) expliquée par le motif U(t) (le reste : bâtiment).

    Seuls les groupes d'au moins `min_size` simulations sont gardés pour que la variance
    intra-groupe soit estimable.
    """
    sizes = np.bincount(groups)
    keep = sizes[groups] >= min_size
    g = groups[keep]
    idx = np.arange(step, temperature.shape[1], step)
    t = np.asarray(temperature)[keep][:, idx].astype(np.float64)
    n_g = np.bincount(g)[:, None]
    means = np.zeros((n_g.shape[0], t.shape[1]))
    np.add.at(means, g, t)
    means = np.divide(means, n_g, out=np.zeros_like(means), where=n_g > 0)
    between = ((means[g] - t.mean(axis=0)) ** 2).sum(axis=0)
    total = ((t - t.mean(axis=0)) ** 2).sum(axis=0)
    return pd.Series(between / total, index=idx * dt_minutes / 60, name="η² (motif U)")


# --------------------------------------------------------------------------- information / temps
def temporal_information_profile(
    temperature: np.ndarray, y: np.ndarray, labels: Sequence[str], step: int = 10,
    dt_minutes: float = 1.0
) -> pd.DataFrame:
    """|Spearman| entre T_int(t) et chaque indicateur, pour t parcourant la fenêtre.

    À appliquer à un groupe de simulations partageant le même U(t) : l'excitation étant fixée,
    les écarts entre courbes ne viennent que du bâtiment (expérience contrôlée).
    """
    t = np.asarray(temperature)
    idx = np.arange(step, t.shape[1], step)
    corr = spearman_matrix(t[:, idx], y)
    return pd.DataFrame(np.abs(corr), index=idx * dt_minutes / 60, columns=list(labels))


def curve_svd(curves: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ACP des courbes (centrées) par SVD.

    Returns:
        Part de variance expliquée (k,), scores (N, k), composantes (k, T).
    """
    c = np.asarray(curves, dtype=np.float64)
    c = c - c.mean(axis=0)
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    explained = s**2 / (s**2).sum()
    return explained[:k], u[:, :k] * s[:k], vt[:k]


# --------------------------------------------------------------------------- cibles
def robust_outliers(
    y: np.ndarray, labels: Sequence[str], threshold: float = 5.0
) -> pd.DataFrame:
    """Valeurs atypiques par z-score robuste (médiane / MAD) sur le log des indicateurs."""
    logy = np.log(np.asarray(y, dtype=np.float64))
    med = np.median(logy, axis=0)
    mad = 1.4826 * np.median(np.abs(logy - med), axis=0)
    z = (logy - med) / mad
    return pd.DataFrame(
        {
            "asymétrie (brut)": pd.DataFrame(y).skew().to_numpy(),
            "asymétrie (log)": pd.DataFrame(logy).skew().to_numpy(),
            "|z robuste| max": np.abs(z).max(axis=0),
            f"part |z| > {threshold:g}": (np.abs(z) > threshold).mean(axis=0),
        },
        index=list(labels),
    )


def snr_table(rise: np.ndarray, noise_std: Sequence[float]) -> pd.DataFrame:
    """Part des simulations dont la montée de T_int est inférieure à k fois le bruit capteur."""
    r = np.asarray(rise)
    rows = {
        f"σ = {s:g} °C": {
            "médiane montée / σ": float(np.median(r) / s),
            "part montée < 3σ": float((r < 3 * s).mean()),
            "part montée < 10σ": float((r < 10 * s).mean()),
        }
        for s in noise_std
    }
    return pd.DataFrame(rows).T
