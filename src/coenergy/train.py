"""Boucle d'entraînement, prédiction et sauvegarde des modèles."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from coenergy.config import ModelConfig, TrainConfig
from coenergy.data import BatchIterator, InputScaler, SimulationData, TargetScaler
from coenergy.losses import multitask_loss
from coenergy.model import CNNGRU


@dataclass
class History:
    """Historique des pertes par époque."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_parts: list[dict[str, float]] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_epoch: int = -1


@torch.no_grad()
def evaluate_loss(
    model: nn.Module, iterator: BatchIterator, target_scaler: TargetScaler, cfg: TrainConfig
) -> tuple[float, dict[str, float]]:
    """Perte moyenne (pondérée par la taille des batchs) sur un itérateur."""
    model.eval()
    total, n = 0.0, 0
    parts_sum: dict[str, float] = {}
    for batch in iterator:
        loss, parts = multitask_loss(model(batch.x), batch, target_scaler, cfg.loss_weights)
        b = batch.x.shape[0]
        total += float(loss) * b
        n += b
        for k, v in parts.items():
            parts_sum[k] = parts_sum.get(k, 0.0) + v * b
    return total / n, {k: v / n for k, v in parts_sum.items()}


def train_model(
    model: nn.Module,
    train_iter: BatchIterator,
    val_iter: BatchIterator,
    target_scaler: TargetScaler,
    cfg: TrainConfig,
) -> History:
    """Entraîne le modèle avec AdamW, réduction du learning rate et early stopping.

    Les poids de la meilleure époque (perte de validation minimale) sont restaurés à la fin.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=cfg.lr_patience
    )
    history = History()
    best_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(cfg.max_epochs):
        model.train()
        running, n = 0.0, 0
        for batch in tqdm(train_iter, desc=f"Époque {epoch + 1}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = multitask_loss(model(batch.x), batch, target_scaler, cfg.loss_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += float(loss) * batch.x.shape[0]
            n += batch.x.shape[0]

        val_loss, val_parts = evaluate_loss(model, val_iter, target_scaler, cfg)
        scheduler.step(val_loss)
        history.train_loss.append(running / n)
        history.val_loss.append(val_loss)
        history.val_parts.append(val_parts)
        history.lr.append(optimizer.param_groups[0]["lr"])
        print(
            f"Époque {epoch + 1:3d} | train {running / n:.4f} | val {val_loss:.4f} | "
            + " ".join(f"{k}={v:.4f}" for k, v in val_parts.items())
        )

        if val_loss < best_loss - 1e-6:
            best_loss, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping à l'époque {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@dataclass
class Predictions:
    """Prédictions en unités physiques, alignées sur `index`."""

    index: np.ndarray
    hlc: np.ndarray
    hb: np.ndarray
    tau: np.ndarray

    def indicators(self) -> np.ndarray:
        """Matrice (N, 1 + n_hb + n_tau) dans l'ordre de `SimulationData.indicators()`."""
        return np.concatenate([self.hlc[:, None], self.hb, self.tau], axis=1)


@torch.no_grad()
def predict(model: nn.Module, iterator: BatchIterator, target_scaler: TargetScaler) -> Predictions:
    """Prédit tous les indicateurs en unités physiques (W/K et h)."""
    model.eval()
    idx, hlc, hb, tau = [], [], [], []
    for batch in iterator:
        out = model(batch.x)
        idx.append(batch.index.cpu())
        hlc.append(target_scaler.decode_hb(out["hlc"]).cpu())
        hb.append(target_scaler.decode_hb(out["hb"]).cpu())
        tau.append(target_scaler.decode_tau(out["tau_z"]).cpu())
    return Predictions(
        index=torch.cat(idx).numpy(),
        hlc=torch.cat(hlc).numpy(),
        hb=torch.cat(hb).numpy(),
        tau=torch.cat(tau).numpy(),
    )


def ground_truth(data: SimulationData, index: np.ndarray) -> np.ndarray:
    """Indicateurs vrais (N, K) pour les simulations `index`."""
    return data.indicators()[torch.as_tensor(index)].numpy()


def save_checkpoint(
    path: str | Path,
    model: CNNGRU,
    input_scaler: InputScaler,
    target_scaler: TargetScaler,
    history: History | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    """Sauvegarde le modèle, sa configuration et les scalers dans un seul fichier."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "input_scaler": input_scaler.state_dict(),
            "target_scaler": target_scaler.state_dict(),
            "history": asdict(history) if history is not None else None,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[CNNGRU, InputScaler, TargetScaler, History | None]:
    """Recharge un modèle sauvegardé avec `save_checkpoint`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    cfg["conv_channels"] = tuple(cfg["conv_channels"])
    model = CNNGRU(ModelConfig(**cfg)).to(device)
    model.load_state_dict(ckpt["model_state"])
    history = History(**ckpt["history"]) if ckpt.get("history") else None
    input_scaler = InputScaler(**ckpt["input_scaler"])
    target_scaler = TargetScaler(**ckpt["target_scaler"])
    return model, input_scaler, target_scaler, history
