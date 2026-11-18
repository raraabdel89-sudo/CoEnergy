"""Chargement des simulations, découpage, normalisation et itération par batchs.

Les entrées du modèle sont construites à la volée, batch par batch, pour limiter la mémoire :
trois canaux sont empilés, T_int, T_int - T_int(0) (montée en température) et U.
"""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F

from coenergy.config import NoiseConfig

INPUT_CHANNELS: list[str] = ["T_int", "delta_T_int", "U"]


@dataclass
class SimulationData:
    """Jeu de simulations en mémoire.

    Attributes:
        temperature: Température intérieure, forme (N, T), en °C.
        excitation: Sollicitation thermique, forme (N, T), en W.
        hlc: Coefficient de déperdition global, forme (N,), en W/K.
        hb: Décomposition du HLC, forme (N, 2), en W/K.
        tau: Constantes de temps, forme (N, 3), en h.
        hb_labels: Noms des composantes de Hb.
        tau_labels: Noms des constantes de temps.
        dt_minutes: Pas de temps des séries, en minutes.
        batch_id: Numéro du fichier batch d'origine de chaque simulation.
        solicitation: Paramètres de l'excitation de chaque simulation, forme (N, P).
        solicitation_names: Noms des paramètres de l'excitation.
    """

    temperature: torch.Tensor
    excitation: torch.Tensor
    hlc: torch.Tensor
    hb: torch.Tensor
    tau: torch.Tensor
    hb_labels: list[str]
    tau_labels: list[str]
    dt_minutes: float = 1.0
    batch_id: torch.Tensor | None = None
    solicitation: torch.Tensor | None = None
    solicitation_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.temperature.shape[0])

    @property
    def n_steps(self) -> int:
        """Nombre de pas de temps par simulation."""
        return int(self.temperature.shape[1])

    @property
    def indicator_labels(self) -> list[str]:
        """Noms des indicateurs dans l'ordre de `indicators()`."""
        return ["HLC", *self.hb_labels, *self.tau_labels]

    def indicators(self) -> torch.Tensor:
        """Vecteur cible complet (HLC, Hb, τ), forme (N, 1 + n_hb + n_tau)."""
        return torch.cat([self.hlc.unsqueeze(1), self.hb, self.tau], dim=1)


def _to_2d(x: object) -> torch.Tensor:
    """Convertit une série temporelle en tenseur (N, T) en retirant un éventuel axe de taille 1."""
    t = torch.as_tensor(x)
    if t.ndim == 3 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    elif t.ndim == 3 and t.shape[1] == 1:
        t = t.squeeze(1)
    if t.ndim != 2:
        raise ValueError(f"Série temporelle attendue de forme (N, T), reçu {tuple(t.shape)}")
    return t


def _downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Moyenne par blocs de `factor` pas de temps (conserve l'énergie injectée moyenne)."""
    if factor <= 1:
        return x
    return F.avg_pool1d(x.unsqueeze(1), kernel_size=factor, stride=factor).squeeze(1)


def _import_util(util_dir: str | Path | None) -> ModuleType:
    """Importe `util_assessement` (qui enregistre aussi le faux paquet nécessaire au unpickling)."""
    if util_dir is not None and str(util_dir) not in sys.path:
        sys.path.insert(0, str(util_dir))
    return importlib.import_module("util_assessement")


def list_batch_files(simulations_path: str | Path) -> list[Path]:
    """Liste triée des fichiers `batch_*.pt` du dossier de simulations."""
    files = sorted(
        Path(simulations_path).glob("batch_*.pt"),
        key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0),
    )
    if not files:
        raise FileNotFoundError(f"Aucun fichier batch_*.pt dans {simulations_path}")
    return files


def describe_batch(
    simulations_path: str | Path, util_dir: str | Path | None = None, batch_index: int = 0
) -> dict[str, object]:
    """Forme, type et libellés de chaque champ d'un fichier batch (pour l'inspection initiale)."""
    util = _import_util(util_dir)
    batch = util.load_single_batch(list_batch_files(simulations_path)[batch_index])
    info: dict[str, object] = {}
    for name in batch.__dataclass_fields__:
        value = getattr(batch, name)
        if isinstance(value, torch.Tensor):
            info[name] = f"{tuple(value.shape)} {value.dtype}"
        else:
            info[name] = list(value)
    return info


def load_solicitation_sample(
    simulations_path: str | Path,
    util_dir: str | Path | None = None,
    batch_index: int = 0,
    n: int = 2000,
    seed: int = 0,
) -> tuple[torch.Tensor, list[str]]:
    """Charge le champ `solicitation` (conditions aux limites) de `n` simulations d'un batch.

    Ce champ n'est pas une entrée imposée par l'énoncé, mais il permet de vérifier si la météo
    et les conditions aux limites varient d'une simulation à l'autre.
    """
    util = _import_util(util_dir)
    batch = util.load_single_batch(list_batch_files(simulations_path)[batch_index])
    sol = torch.as_tensor(batch.solicitation)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(sol.shape[0], generator=gen)[:n].sort().values
    return sol[idx].float(), [str(name) for name in batch.solicitation_field_names]


def load_simulations(
    simulations_path: str | Path,
    util_dir: str | Path | None = None,
    n_batches: int | None = None,
    n_per_batch: int | None = None,
    downsample: int = 1,
    seed: int = 42,
) -> SimulationData:
    """Charge les simulations batch par batch pour maîtriser la mémoire.

    Contrairement à `util_assessement.load_simulation_data`, qui charge et concatène tous les
    batchs, on ne garde que les champs utiles des `n_batches` premiers fichiers, et
    éventuellement un tirage aléatoire de `n_per_batch` simulations dans chacun.

    Args:
        simulations_path: Dossier contenant les fichiers `batch_*.pt`.
        util_dir: Dossier contenant `util_assessement.py` (ajouté au `sys.path`).
        n_batches: Nombre de fichiers batch à charger (None = tous).
        n_per_batch: Nombre de simulations tirées par batch (None = toutes).
        downsample: Facteur de sous-échantillonnage temporel.
        seed: Graine du tirage.

    Returns:
        Les simulations en tenseurs float32.
    """
    util = _import_util(util_dir)
    files = list_batch_files(simulations_path)
    if n_batches is not None:
        files = files[:n_batches]

    gen = torch.Generator().manual_seed(seed)
    parts: dict[str, list[torch.Tensor]] = {
        k: [] for k in ["temperature", "excitation", "hlc", "hb", "tau", "batch_id", "sol"]
    }
    hb_labels: list[str] = []
    sol_names: list[str] = []
    for i, path in enumerate(files):
        batch = util.load_single_batch(path)
        temperature = _to_2d(batch.indoor_temperature)
        n = temperature.shape[0]
        if n_per_batch is not None and n_per_batch < n:
            idx = torch.randperm(n, generator=gen)[:n_per_batch].sort().values
        else:
            idx = torch.arange(n)
        excitation = _to_2d(batch.indoor_U)[idx].round(decimals=1).float()
        parts["temperature"].append(
            _downsample(temperature[idx].round(decimals=2).float(), downsample)
        )
        parts["excitation"].append(_downsample(excitation, downsample))
        parts["hlc"].append(torch.as_tensor(batch.HLC).reshape(-1)[idx].float())
        parts["hb"].append(torch.as_tensor(batch.Hb)[idx].float())
        parts["tau"].append(torch.as_tensor(batch.CDT)[idx].float())
        parts["batch_id"].append(torch.full((len(idx),), i, dtype=torch.int16))
        parts["sol"].append(torch.as_tensor(batch.solicitation)[idx].float())
        hb_labels = [str(label) for label in batch.Hb_labels]
        sol_names = [str(name) for name in batch.solicitation_field_names]
        del batch
        print(f"{path.name} : {len(idx)} simulations conservées")

    return SimulationData(
        temperature=torch.cat(parts["temperature"]),
        excitation=torch.cat(parts["excitation"]),
        hlc=torch.cat(parts["hlc"]),
        hb=torch.cat(parts["hb"]),
        tau=torch.cat(parts["tau"]),
        hb_labels=hb_labels,
        tau_labels=["tau_b", "tau_n", "tau_inf"],
        dt_minutes=float(max(downsample, 1)),
        batch_id=torch.cat(parts["batch_id"]),
        solicitation=torch.cat(parts["sol"]),
        solicitation_names=sol_names,
    )


def make_splits(
    n: int, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42
) -> dict[str, torch.Tensor]:
    """Découpe aléatoire et déterministe des indices en train / val / test.

    Le découpage se fait par simulation : une simulation n'apparaît que dans un seul ensemble.
    """
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
    return {
        "test": perm[:n_test].sort().values,
        "val": perm[n_test : n_test + n_val].sort().values,
        "train": perm[n_test + n_val :].sort().values,
    }


def build_inputs(temperature: torch.Tensor, excitation: torch.Tensor) -> torch.Tensor:
    """Empile les canaux d'entrée du modèle.

    Args:
        temperature: (B, T) en °C.
        excitation: (B, T) en W.

    Returns:
        Tenseur (B, 3, T) : T_int, T_int - T_int(0), U.
    """
    delta = temperature - temperature[:, :1]
    return torch.stack([temperature, delta, excitation], dim=1)


@dataclass
class InputScaler:
    """Standardisation par canal, ajustée sur le jeu d'entraînement."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, data: SimulationData, indices: torch.Tensor, chunk: int = 8192) -> InputScaler:
        """Calcule moyenne et écart-type de chaque canal, par morceaux pour limiter la mémoire."""
        s = torch.zeros(len(INPUT_CHANNELS), dtype=torch.float64)
        s2 = torch.zeros_like(s)
        count = 0
        for start in range(0, len(indices), chunk):
            idx = indices[start : start + chunk]
            x = build_inputs(data.temperature[idx], data.excitation[idx]).double()
            s += x.sum(dim=(0, 2))
            s2 += (x**2).sum(dim=(0, 2))
            count += x.shape[0] * x.shape[2]
        mean = s / count
        std = (s2 / count - mean**2).clamp_min(1e-12).sqrt()
        return cls(mean=mean.float(), std=std.float())

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Applique la standardisation à un batch (B, C, T)."""
        mean = self.mean.to(x.device)[None, :, None]
        std = self.std.to(x.device)[None, :, None]
        return (x - mean) / std

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Sérialisation pour le checkpoint."""
        return {"mean": self.mean, "std": self.std}


@dataclass
class TargetScaler:
    """Mise à l'échelle des cibles.

    - Hb et HLC sont divisés par une constante (le HLC moyen) : ils restent positifs, et la
      relation HLC = somme(Hb) est conservée dans l'espace normalisé.
    - Les τ sont passées en log puis standardisées (distribution positive et asymétrique).
    - `hb_std` et `hlc_std` servent à normaliser la perte pour équilibrer les familles.
    """

    hb_scale: float
    hb_std: torch.Tensor
    hlc_std: float
    log_tau_mean: torch.Tensor
    log_tau_std: torch.Tensor

    @classmethod
    def fit(cls, data: SimulationData, indices: torch.Tensor) -> TargetScaler:
        """Ajuste les paramètres d'échelle sur le jeu d'entraînement."""
        scale = float(data.hlc[indices].mean())
        log_tau = torch.log(data.tau[indices].clamp_min(1e-6))
        return cls(
            hb_scale=scale,
            hb_std=(data.hb[indices] / scale).std(dim=0).clamp_min(1e-6),
            hlc_std=float((data.hlc[indices] / scale).std().clamp_min(1e-6)),
            log_tau_mean=log_tau.mean(dim=0),
            log_tau_std=log_tau.std(dim=0).clamp_min(1e-6),
        )

    def encode_hb(self, hb: torch.Tensor) -> torch.Tensor:
        """W/K → espace normalisé (s'applique aussi au HLC)."""
        return hb / self.hb_scale

    def decode_hb(self, z: torch.Tensor) -> torch.Tensor:
        """Espace normalisé → W/K."""
        return z * self.hb_scale

    def encode_tau(self, tau: torch.Tensor) -> torch.Tensor:
        """Heures → log standardisé."""
        m, s = self.log_tau_mean.to(tau.device), self.log_tau_std.to(tau.device)
        return (torch.log(tau.clamp_min(1e-6)) - m) / s

    def decode_tau(self, z: torch.Tensor) -> torch.Tensor:
        """Log standardisé → heures."""
        m, s = self.log_tau_mean.to(z.device), self.log_tau_std.to(z.device)
        return torch.exp(z * s + m)

    def state_dict(self) -> dict[str, object]:
        """Sérialisation pour le checkpoint."""
        return {
            "hb_scale": self.hb_scale,
            "hb_std": self.hb_std,
            "hlc_std": self.hlc_std,
            "log_tau_mean": self.log_tau_mean,
            "log_tau_std": self.log_tau_std,
        }


@dataclass
class Batch:
    """Batch prêt pour le modèle (cibles dans l'espace normalisé)."""

    x: torch.Tensor
    hb: torch.Tensor
    hlc: torch.Tensor
    tau_z: torch.Tensor
    index: torch.Tensor


class BatchIterator:
    """Itère sur un sous-ensemble de simulations en construisant les batchs à la volée.

    Le bruit capteur (si fourni) est tiré à chaque passage : deux époques voient donc deux
    réalisations différentes du bruit, ce qui agit comme une augmentation de données.
    """

    def __init__(
        self,
        data: SimulationData,
        indices: torch.Tensor,
        input_scaler: InputScaler,
        target_scaler: TargetScaler,
        batch_size: int = 256,
        shuffle: bool = False,
        noise: NoiseConfig | None = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.data = data
        self.indices = indices
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise = noise
        self.device = torch.device(device)
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self) -> Iterator[Batch]:
        order = self.indices
        if self.shuffle:
            self._epoch += 1
            gen = torch.Generator().manual_seed(self.seed + self._epoch)
            order = order[torch.randperm(len(order), generator=gen)]
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            temperature = self.data.temperature[idx].to(self.device)
            excitation = self.data.excitation[idx].to(self.device)
            if self.noise is not None:
                temperature = temperature + self.noise.temperature_std * torch.randn_like(
                    temperature
                )
                excitation = excitation * (
                    1 + self.noise.excitation_rel_std * torch.randn_like(excitation)
                )
            x = self.input_scaler.transform(build_inputs(temperature, excitation))
            ts = self.target_scaler
            yield Batch(
                x=x,
                hb=ts.encode_hb(self.data.hb[idx].to(self.device)),
                hlc=ts.encode_hb(self.data.hlc[idx].to(self.device)),
                tau_z=ts.encode_tau(self.data.tau[idx].to(self.device)),
                index=idx,
            )
