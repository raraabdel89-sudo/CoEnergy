"""Configuration des expériences sous forme de dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Paramètres de chargement et de découpage des données.

    Attributes:
        n_batches: Nombre de fichiers batch chargés (None = les 20, soit 1 000 000 simulations).
        n_per_batch: Simulations tirées aléatoirement dans chaque batch (None = les 50 000).
        downsample: Facteur de sous-échantillonnage temporel (moyenne glissante sans recouvrement).
        val_frac: Fraction des simulations réservée à la validation.
        test_frac: Fraction des simulations réservée au test final.
        seed: Graine du sous-échantillonnage et du découpage.
    """

    n_batches: int | None = 4
    n_per_batch: int | None = None
    downsample: int = 1
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42


@dataclass
class NoiseConfig:
    """Bruit gaussien ajouté aux entrées pour imiter les capteurs réels.

    Attributes:
        temperature_std: Écart-type du bruit additif sur T_int (°C).
        excitation_rel_std: Écart-type du bruit multiplicatif relatif sur U (sans unité).
    """

    temperature_std: float = 0.1
    excitation_rel_std: float = 0.01


@dataclass
class ModelConfig:
    """Hyperparamètres de l'architecture CNN-GRU."""

    in_channels: int = 3
    conv_channels: tuple[int, ...] = (64, 64, 128)
    kernel_size: int = 7
    hidden_size: int = 128
    num_gru_layers: int = 2
    dropout: float = 0.1
    n_hb: int = 2
    n_tau: int = 3


@dataclass
class TrainConfig:
    """Hyperparamètres d'entraînement."""

    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 40
    patience: int = 6
    lr_patience: int = 2
    grad_clip: float = 1.0
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"hb": 1.0, "hlc": 1.0, "tau": 1.0}
    )
    seed: int = 42
