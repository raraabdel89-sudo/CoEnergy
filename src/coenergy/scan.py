"""Parcours en flux de TOUS les fichiers batch pour l'analyse exploratoire.

Charger 1 000 000 de simulations en mémoire (2 séries de 480 pas) demande près de 4 Go en
float32, et le double pour les calculs en float64. On parcourt donc les fichiers un par un et
l'on ne garde que :

- pour **chaque** simulation : indicateurs, paramètres d'excitation, descripteurs scalaires
  des séries et empreintes (hash) des signaux, dans une table pandas ;
- un **échantillon aléatoire de courbes**, tiré uniformément dans tous les batchs, pour les
  analyses qui exigent les séries complètes (spectre, ACP, identification R1C1, enveloppes) ;
- des **agrégats par groupe d'excitation** (second passage) : variance expliquée η²(t) et
  courbes des groupes d'« expérience contrôlée » (même U(t) pour toutes les simulations).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from coenergy.data import _import_util, _to_2d, list_batch_files
from coenergy.eda import excitation_richness
from coenergy.features import excitation_descriptors, temperature_descriptors

STEP_CATEGORY = "un seul front (échelon)"
ZERO_CATEGORY = "aucune chauffe"


def row_hashes(x: np.ndarray) -> np.ndarray:
    """Empreinte 64 bits de chaque ligne (égalité exacte des octets)."""
    rows = np.ascontiguousarray(x).view(np.uint8).reshape(len(x), -1)
    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), "little",
                        signed=True) for r in rows),
        dtype=np.int64, count=len(rows),
    )


@dataclass
class FullScan:
    """Résumé de l'ensemble des simulations (voir le docstring du module)."""

    table: pd.DataFrame
    labels: list[str]
    solicitation_names: list[str]
    dt_minutes: float
    n_steps: int
    files: list[str]
    checks: dict[str, float]
    sample_idx: np.ndarray
    sample_temperature: np.ndarray
    sample_excitation: np.ndarray
    u_groups: np.ndarray
    eta2: pd.Series | None = None
    controlled_idx: dict[str, np.ndarray] = field(default_factory=dict)
    controlled_temperature: dict[str, np.ndarray] = field(default_factory=dict)
    controlled_excitation: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.table)

    @property
    def y(self) -> np.ndarray:
        """Indicateurs (N, 6) dans l'ordre de `labels`."""
        return self.table[self.labels].to_numpy(dtype=np.float64)

    @property
    def solicitation(self) -> pd.DataFrame:
        return self.table[self.solicitation_names]

    @property
    def sample_y(self) -> np.ndarray:
        return self.y[self.sample_idx]


def _read_batch(util, path: Path) -> dict[str, np.ndarray]:
    """Lit un fichier avec le même prétraitement que `load_simulations` (arrondis)."""
    batch = util.load_single_batch(path)
    out = {
        "temperature": _to_2d(batch.indoor_temperature).round(decimals=2).float().numpy(),
        "excitation": _to_2d(batch.indoor_U).round(decimals=1).float().numpy(),
        "hlc": torch.as_tensor(batch.HLC).reshape(-1).float().numpy(),
        "hb": torch.as_tensor(batch.Hb).float().numpy(),
        "tau": torch.as_tensor(batch.CDT).float().numpy(),
        "sol": torch.as_tensor(batch.solicitation).float().numpy(),
        "hb_labels": [str(x) for x in batch.Hb_labels],
        "sol_names": [str(x) for x in batch.solicitation_field_names],
    }
    del batch
    return out


def scan_simulations(
    simulations_path: str | Path,
    util_dir: str | Path | None = None,
    n_batches: int | None = None,
    curves_per_batch: int = 5_000,
    eta_step: int = 20,
    max_controlled: int = 50_000,
    seed: int = 42,
    verbose: bool = True,
) -> FullScan:
    """Parcourt tous les fichiers batch (deux passages) et construit un `FullScan`.

    Args:
        simulations_path: Dossier des fichiers `batch_*.pt`.
        util_dir: Dossier de `util_assessement.py`.
        n_batches: Nombre de fichiers (None = tous).
        curves_per_batch: Courbes conservées par fichier pour l'échantillon.
        eta_step: Pas (en minutes) des instants où η²(t) est calculé.
        max_controlled: Nombre maximal de courbes conservées par groupe contrôlé.
        seed: Graine des tirages.
    """
    util = _import_util(util_dir)
    files = list_batch_files(simulations_path)
    if n_batches is not None:
        files = files[:n_batches]
    rng = np.random.default_rng(seed)
    dt_minutes = 1.0

    # ------------------------------------------------------------ passage 1 : scalaires
    tables, samples_t, samples_u, sample_idx = [], [], [], []
    n_nan, u_min, offset, n_steps = 0, np.inf, 0, 0
    labels: list[str] = []
    sol_names: list[str] = []
    for path in files:
        b = _read_batch(util, path)
        t, u = b["temperature"], b["excitation"]
        n, n_steps = t.shape
        labels = ["HLC", *b["hb_labels"], "tau_b", "tau_n", "tau_inf"]
        sol_names = b["sol_names"]
        n_nan += int(np.isnan(t).sum() + np.isnan(u).sum())
        u_min = min(u_min, float(u.min()))
        exc = excitation_descriptors(u, dt_minutes).drop(columns=["frac_extreme", "kind"])
        tmp = temperature_descriptors(t)
        rich = excitation_richness(u, dt_minutes).drop(columns=["n_switches", "duty_cycle"])
        y = np.concatenate([b["hlc"][:, None], b["hb"], b["tau"]], axis=1)
        part = pd.concat(
            [pd.DataFrame(y, columns=labels), pd.DataFrame(b["sol"], columns=sol_names),
             exc, tmp, rich], axis=1)
        part.insert(0, "batch", int("".join(c for c in path.stem if c.isdigit())))
        part["u_hash"] = row_hashes(u)
        part["t_hash"] = row_hashes(t)
        tables.append(part)
        pick = np.sort(rng.choice(n, size=min(curves_per_batch, n), replace=False))
        samples_t.append(t[pick])
        samples_u.append(u[pick])
        sample_idx.append(offset + pick)
        offset += n
        if verbose:
            print(f"[1/2] {path.name} : {n} simulations")
        del b, t, u

    table = pd.concat(tables, ignore_index=True)
    table["categorie"] = table["categorie"].astype("category")
    u_groups = pd.factorize(table["u_hash"])[0]
    y_all = table[labels].to_numpy()
    checks = {
        "simulations": float(len(table)),
        "valeurs manquantes (entrées)": float(n_nan),
        "valeurs manquantes (cibles)": float(np.isnan(y_all).sum()),
        "indicateurs <= 0": float((y_all <= 0).sum()),
        "max |HLC - (HLC_ext + HLC_mit)| (W/K)": float(
            np.abs(y_all[:, 0] - y_all[:, 1] - y_all[:, 2]).max()),
        "part de tau_n <= tau_inf": float((y_all[:, 4] <= y_all[:, 5]).mean()),
        "part de tau_b = max des tau": float((y_all[:, 3:].argmax(axis=1) == 0).mean()),
        "U minimale (W)": u_min,
    }

    # groupes d'expérience contrôlée : signal nul et échelon le plus fréquent
    cat = table["categorie"].to_numpy()
    controlled_groups = {}
    if (cat == ZERO_CATEGORY).any():
        controlled_groups[ZERO_CATEGORY] = int(u_groups[np.argmax(cat == ZERO_CATEGORY)])
    if (cat == STEP_CATEGORY).any():
        controlled_groups[STEP_CATEGORY] = int(np.bincount(u_groups[cat == STEP_CATEGORY]).argmax())
    keep: dict[str, np.ndarray] = {}
    for name, g in controlled_groups.items():
        members = np.where(u_groups == g)[0]
        if len(members) > max_controlled:
            members = np.sort(rng.choice(members, size=max_controlled, replace=False))
        keep[name] = members

    # ------------------------------------------------------------ passage 2 : groupes
    t_idx = np.arange(eta_step, n_steps, eta_step)
    n_groups = int(u_groups.max()) + 1
    sums = np.zeros((n_groups, len(t_idx)))
    sumsq = np.zeros((n_groups, len(t_idx)))
    counts = np.bincount(u_groups, minlength=n_groups).astype(np.float64)
    collected = {name: [] for name in keep}
    excitation_of: dict[str, np.ndarray] = {}
    offset = 0
    for path in files:
        b = _read_batch(util, path)
        t = b["temperature"]
        n = len(t)
        g = u_groups[offset:offset + n]
        sel = t[:, t_idx].astype(np.float64)
        np.add.at(sums, g, sel)
        np.add.at(sumsq, g, sel**2)
        for name, members in keep.items():
            local = members[(members >= offset) & (members < offset + n)] - offset
            collected[name].append(t[local])
            if len(local) and name not in excitation_of:
                excitation_of[name] = b["excitation"][local[0]].copy()
        offset += n
        if verbose:
            print(f"[2/2] {path.name}")
        del b, t

    # η² sur les groupes d'au moins 2 simulations (variance intra-groupe estimable)
    ok = counts >= 2
    sums, sumsq, counts = sums[ok], sumsq[ok], counts[ok]
    total_mean = sums.sum(axis=0) / counts.sum()
    group_means = sums / counts[:, None]
    between = (counts[:, None] * (group_means - total_mean) ** 2).sum(axis=0)
    total = sumsq.sum(axis=0) - counts.sum() * total_mean**2
    eta2 = pd.Series(between / total, index=t_idx * dt_minutes / 60, name="η² (motif U)")

    return FullScan(
        table=table, labels=labels, solicitation_names=sol_names, dt_minutes=dt_minutes,
        n_steps=n_steps, files=[p.name for p in files], checks=checks,
        sample_idx=np.concatenate(sample_idx), sample_temperature=np.concatenate(samples_t),
        sample_excitation=np.concatenate(samples_u), u_groups=u_groups, eta2=eta2,
        controlled_idx=keep,
        controlled_temperature={k: np.concatenate(v) for k, v in collected.items()},
        controlled_excitation=excitation_of,
    )
