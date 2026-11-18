"""Figures de l'analyse exploratoire et de l'évaluation (matplotlib)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

UNITS: dict[str, str] = {"HLC": "W/K", "CDT": "h", "Hb": "W/K"}


def _unit(label: str) -> str:
    return "h" if label.startswith("tau") else "W/K"


def plot_target_distributions(y: np.ndarray, labels: Sequence[str], bins: int = 80) -> Figure:
    """Histogrammes de chaque indicateur."""
    k = len(labels)
    fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(4 * ((k + 1) // 2), 6.5))
    for ax, j, label in zip(axes.flat, range(k), labels, strict=False):
        ax.hist(y[:, j], bins=bins, color="#534AB7", alpha=0.85)
        ax.set_title(label)
        ax.set_xlabel(_unit(label))
    for ax in list(axes.flat)[k:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_correlation_matrix(
    values: np.ndarray, labels: Sequence[str], method: str = "spearman"
) -> Figure:
    """Matrice de corrélation (Spearman par défaut : robuste aux relations non linéaires)."""
    data = np.asarray(values, dtype=np.float64)
    if method == "spearman":
        data = np.argsort(np.argsort(data, axis=0), axis=0).astype(np.float64)
    corr = np.corrcoef(data, rowvar=False)
    fig, ax = plt.subplots(figsize=(1 + 0.8 * len(labels), 0.8 * len(labels)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Corrélation ({method})")
    fig.tight_layout()
    return fig


def plot_sample_series(
    temperature: np.ndarray,
    excitation: np.ndarray,
    dt_minutes: float = 1.0,
    titles: Sequence[str] | None = None,
) -> Figure:
    """Trace quelques couples (T_int, U) l'un au-dessus de l'autre."""
    n = temperature.shape[0]
    t = np.arange(temperature.shape[1]) * dt_minutes / 60.0
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 5), sharex=True)
    axes = np.atleast_2d(axes).reshape(2, n)
    for i in range(n):
        axes[0, i].plot(t, temperature[i], color="#D85A30")
        axes[1, i].plot(t, excitation[i], color="#534AB7")
        axes[0, i].set_title(titles[i] if titles else f"Simulation {i}", fontsize=10)
        axes[1, i].set_xlabel("Temps (h)")
    axes[0, 0].set_ylabel("T_int (°C)")
    axes[1, 0].set_ylabel("U (W)")
    fig.tight_layout()
    return fig


def plot_learning_curves(train_loss: Sequence[float], val_loss: Sequence[float]) -> Figure:
    """Courbes de perte d'entraînement et de validation."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_loss, label="train")
    ax.plot(val_loss, label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Perte normalisée")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_parity(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str], max_points: int = 20_000
) -> Figure:
    """Prédiction en fonction de la vérité, un panneau par indicateur."""
    rng = np.random.default_rng(0)
    sel = rng.choice(len(y_true), size=min(max_points, len(y_true)), replace=False)
    k = len(labels)
    fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(4 * ((k + 1) // 2), 7))
    for ax, j, label in zip(axes.flat, range(k), labels, strict=False):
        yt, yp = y_true[sel, j], y_pred[sel, j]
        ax.scatter(yt, yp, s=2, alpha=0.3, color="#1D9E75")
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(label)
        ax.set_xlabel(f"Vrai ({_unit(label)})")
        ax.set_ylabel("Prédit")
    for ax in list(axes.flat)[k:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_error_vs_value(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str], n_bins: int = 10
) -> Figure:
    """Erreur relative absolue médiane et P95 par décile de la valeur vraie."""
    k = len(labels)
    fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(4 * ((k + 1) // 2), 7))
    for ax, j, label in zip(axes.flat, range(k), labels, strict=False):
        yt = y_true[:, j]
        rel = 100 * np.abs(y_pred[:, j] - yt) / np.maximum(np.abs(yt), 1e-9)
        edges = np.quantile(yt, np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.digitize(yt, edges[1:-1]), 0, n_bins - 1)
        centers = [np.median(yt[bins == b]) for b in range(n_bins)]
        med = [np.median(rel[bins == b]) for b in range(n_bins)]
        p95 = [np.percentile(rel[bins == b], 95) for b in range(n_bins)]
        ax.plot(centers, med, "o-", label="médiane")
        ax.plot(centers, p95, "s--", label="P95")
        ax.set_title(label)
        ax.set_xlabel(f"Valeur vraie ({_unit(label)})")
        ax.set_ylabel("Erreur relative (%)")
    axes.flat[0].legend()
    for ax in list(axes.flat)[k:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- EDA approfondie
def plot_log_distributions(y: np.ndarray, labels: Sequence[str], bins: int = 80) -> Figure:
    """Histogramme du log de chaque indicateur (densité), avec la loi normale ajustée,
    et QQ-plot des log-valeurs contre une loi normale (ligne du bas)."""
    from statistics import NormalDist

    logy = np.log(np.asarray(y, dtype=np.float64))
    k = len(labels)
    fig, axes = plt.subplots(2, k, figsize=(3.2 * k, 6))
    probs = np.linspace(0.001, 0.999, 300)
    theo = np.array([NormalDist().inv_cdf(p) for p in probs])
    for j, label in enumerate(labels):
        x = logy[:, j]
        mu, sd = x.mean(), x.std()
        ax = axes[0, j]
        ax.hist(x, bins=bins, density=True, color="#534AB7", alpha=0.75)
        grid = np.linspace(x.min(), x.max(), 200)
        ax.plot(grid, np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
                "k-", lw=1.2)
        ax.set_title(f"log {label}")
        ax = axes[1, j]
        ax.plot(theo, (np.quantile(x, probs) - mu) / sd, ".", ms=2, color="#D85A30")
        ax.plot([-3.5, 3.5], [-3.5, 3.5], "k--", lw=1)
        ax.set_xlabel("quantile normal théorique")
        if j == 0:
            ax.set_ylabel("quantile observé (log standardisé)")
    fig.suptitle("Distributions en échelle log (haut) et QQ-plots contre une loi normale (bas)")
    fig.tight_layout()
    return fig


def plot_pair_hexbin(
    y: np.ndarray, labels: Sequence[str], max_points: int = 50_000, gridsize: int = 40
) -> Figure:
    """Matrice de nuages de densité (hexbin) entre indicateurs, histogrammes sur la diagonale."""
    rng = np.random.default_rng(0)
    sel = rng.choice(len(y), size=min(max_points, len(y)), replace=False)
    v = np.asarray(y)[sel]
    k = len(labels)
    fig, axes = plt.subplots(k, k, figsize=(2.3 * k, 2.3 * k))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if i == j:
                ax.hist(v[:, j], bins=50, color="#534AB7")
            elif i > j:
                ax.hexbin(v[:, j], v[:, i], gridsize=gridsize, bins="log", cmap="viridis")
            else:
                ax.set_visible(False)
            ax.tick_params(labelsize=7)
            if i == k - 1:
                ax.set_xlabel(labels[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=9)
    fig.suptitle("Relations deux à deux entre indicateurs (densité en échelle log)")
    fig.tight_layout()
    return fig


def plot_overlaid_distributions(
    y: np.ndarray, labels: Sequence[str], groups: np.ndarray, bins: int = 60
) -> Figure:
    """Histogrammes superposés (contours, densité) d'un indicateur par groupe."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    k = len(labels)
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3))
    for j, (ax, label) in enumerate(zip(axes, labels, strict=True)):
        edges = np.quantile(y[:, j], np.linspace(0.001, 0.999, bins + 1))
        for g in np.unique(groups):
            ax.hist(y[groups == g, j], bins=edges, density=True, histtype="step",
                    label=str(g))
        ax.set_title(label)
    axes[0].legend(fontsize=7, title="groupe", title_fontsize=7)
    fig.tight_layout()
    return fig


def plot_series_envelope(
    temperature: np.ndarray, excitation: np.ndarray, dt_minutes: float = 1.0,
    max_points: int = 20_000
) -> Figure:
    """Enveloppes temporelles : quantiles de T_int(t) et de ΔT(t), probabilité de chauffe P(U>0)."""
    rng = np.random.default_rng(0)
    sel = rng.choice(len(temperature), size=min(max_points, len(temperature)), replace=False)
    t = np.asarray(temperature)[sel].astype(np.float64)
    u = np.asarray(excitation)[sel]
    hours = np.arange(t.shape[1]) * dt_minutes / 60
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    for ax, series, title in [(axes[0], t, "T_int(t) (°C)"),
                              (axes[1], t - t[:, :1], "ΔT_int(t) = T_int - T_int(0) (K)")]:
        q = np.quantile(series, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        ax.fill_between(hours, q[0], q[4], color="#D85A30", alpha=0.2, label="P5–P95")
        ax.fill_between(hours, q[1], q[3], color="#D85A30", alpha=0.4, label="P25–P75")
        ax.plot(hours, q[2], color="#D85A30", label="médiane")
        ax.set_title(title)
        ax.set_xlabel("t (h)")
    axes[0].legend(fontsize=8)
    axes[2].plot(hours, (u > 0).mean(axis=0), color="#534AB7")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Part des simulations en chauffe P(U(t) > 0)")
    axes[2].set_xlabel("t (h)")
    fig.tight_layout()
    return fig


def plot_conditional_curves(
    temperature: np.ndarray, y: np.ndarray, labels: Sequence[str], columns: Sequence[int],
    dt_minutes: float = 1.0, n_bins: int = 5
) -> Figure:
    """Courbe ΔT_int(t) moyenne par quintile de chaque indicateur choisi.

    À appliquer à des simulations de même excitation : les écarts entre courbes sont alors
    attribuables au seul indicateur (et aux paramètres qui lui sont corrélés).
    """
    t = np.asarray(temperature, dtype=np.float64)
    delta = t - t[:, :1]
    hours = np.arange(t.shape[1]) * dt_minutes / 60
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(columns), figsize=(4.6 * len(columns), 3.8), sharey=True)
    for ax, j in zip(np.atleast_1d(axes), columns, strict=True):
        edges = np.quantile(y[:, j], np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.digitize(y[:, j], edges[1:-1]), 0, n_bins - 1)
        for b in range(n_bins):
            m = bins == b
            ax.plot(hours, delta[m].mean(axis=0), color=cmap(b / max(n_bins - 1, 1)),
                    label=f"Q{b + 1} (méd. {np.median(y[m, j]):.0f} {_unit(labels[j])})")
        ax.set_title(f"ΔT_int moyen par quintile de {labels[j]}")
        ax.set_xlabel("t (h)")
        ax.legend(fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("ΔT_int (K)")
    fig.tight_layout()
    return fig


def plot_batch_means(y: np.ndarray, labels: Sequence[str], batch: np.ndarray) -> Figure:
    """Moyenne de chaque indicateur par batch, avec son intervalle de confiance à 95 %.

    La ligne horizontale est la moyenne globale, la bande grise ±1 % autour d'elle.
    """
    y = np.asarray(y, dtype=np.float64)
    batch = np.asarray(batch)
    ids = np.unique(batch)
    k = len(labels)
    fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(4.5 * ((k + 1) // 2), 6), sharex=True)
    for ax, j, label in zip(axes.flat, range(k), labels, strict=False):
        means = np.array([y[batch == b, j].mean() for b in ids])
        half = np.array([1.96 * y[batch == b, j].std() / np.sqrt((batch == b).sum()) for b in ids])
        overall = y[:, j].mean()
        ax.axhspan(0.99 * overall, 1.01 * overall, color="0.9")
        ax.axhline(overall, color="k", lw=0.8)
        ax.errorbar(ids, means, yerr=half, fmt="o", ms=4, color="#534AB7")
        ax.set_title(f"{label} ({_unit(label)})")
        ax.set_xticks(ids)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("batch")
    fig.suptitle("Moyenne par batch ± IC 95 % (bande grise : ±1 % de la moyenne globale)")
    fig.tight_layout()
    return fig
