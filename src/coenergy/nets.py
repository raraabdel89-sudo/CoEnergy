"""Réseaux PyTorch pour la phase d'optimisation : MLP, ResMLP (tabulaire), encodeurs de
séquences (CNN-GRU, TCN, CNN) et modèle hybride séquence + descripteurs.

Tous les modèles prédisent z (5 composantes standardisées, voir `targets.py`).
`fit_torch` entraîne n'importe lequel de ces modèles sur des tenseurs déjà placés sur le GPU,
ce qui évite le coût des DataLoader et rend chaque époque très rapide.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------------------
# Briques
# --------------------------------------------------------------------------------------
ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "leaky_relu": lambda: nn.LeakyReLU(0.05),
    "tanh": nn.Tanh,
    "mish": nn.Mish,
}


def make_activation(name: str) -> nn.Module:
    """Instancie une fonction d'activation par son nom."""
    return ACTIVATIONS[name]()


def make_norm(kind: str, dim: int) -> nn.Module:
    """Normalisation : "batch", "layer" ou "none"."""
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    if kind == "layer":
        return nn.LayerNorm(dim)
    return nn.Identity()


def init_weights(model: nn.Module, scheme: str = "default") -> nn.Module:
    """Initialise les couches linéaires et convolutives : default, kaiming, xavier, orthogonal."""
    if scheme == "default":
        return model
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if scheme == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif scheme == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif scheme == "orthogonal":
                nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


class MLP(nn.Module):
    """Perceptron multicouche : [Linear → Norm → Act → Dropout] × profondeur → Linear."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: list[int],
        activation: str = "gelu",
        dropout: float = 0.1,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [
                nn.Linear(d, h),
                make_norm(norm, h),
                make_activation(activation),
                nn.Dropout(dropout),
            ]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)
        self.out_features = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class ResBlock(nn.Module):
    """Bloc résiduel tabulaire : x + Dropout(Linear(Act(Norm(Linear(Norm(x))))))."""

    def __init__(
        self, width: int, activation: str, dropout: float, norm: str, expansion: int = 2
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            make_norm(norm, width),
            nn.Linear(width, width * expansion),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(width * expansion, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResMLP(nn.Module):
    """MLP résiduel pour données tabulaires (à la Gorishniy et al., 2021)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 256,
        n_blocks: int = 3,
        activation: str = "gelu",
        dropout: float = 0.1,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.stem = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(
            *[ResBlock(width, activation, dropout, norm) for _ in range(n_blocks)]
        )
        self.final = nn.Sequential(make_norm(norm, width), make_activation(activation))
        self.head = nn.Linear(width, out_dim)
        self.out_features = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.final(self.blocks(self.stem(x))))


# --------------------------------------------------------------------------------------
# Encodeurs de séquences
# --------------------------------------------------------------------------------------
class TemporalBlock(nn.Module):
    """Bloc TCN : deux convolutions dilatées (même longueur) avec connexion résiduelle."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class SeqEncoder(nn.Module):
    """Encode une séquence (B, C, L) en un vecteur (B, out_features).

    kind :
    - "cnn_gru" : blocs Conv-BN-GELU-MaxPool puis GRU (état final + moyenne temporelle) ;
    - "tcn" : convolutions dilatées résiduelles (champ réceptif exponentiel), pooling moyen + max ;
    - "cnn" : blocs convolutifs puis pooling moyen + max.
    """

    def __init__(
        self,
        kind: str = "cnn_gru",
        in_ch: int = 3,
        channels: int = 64,
        n_conv: int = 3,
        kernel: int = 5,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "tcn":
            blocks, c = [], in_ch
            for i in range(n_conv):
                blocks.append(TemporalBlock(c, channels, kernel, 2**i, dropout))
                c = channels
            self.conv = nn.Sequential(*blocks)
            self.out_features = 2 * channels
        else:
            blocks, c = [], in_ch
            for _ in range(n_conv):
                blocks += [
                    nn.Conv1d(c, channels, kernel, padding=kernel // 2),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.MaxPool1d(2),
                ]
                c = channels
            self.conv = nn.Sequential(*blocks)
            if kind == "cnn_gru":
                self.gru = nn.GRU(
                    channels,
                    gru_hidden,
                    num_layers=gru_layers,
                    batch_first=True,
                    dropout=dropout if gru_layers > 1 else 0.0,
                )
                self.out_features = 2 * gru_hidden
            else:
                self.out_features = 2 * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        if self.kind == "cnn_gru":
            out, h_n = self.gru(h.transpose(1, 2))
            return torch.cat([h_n[-1], out.mean(dim=1)], dim=1)
        return torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)


class HybridNet(nn.Module):
    """Modèle hybride : encodeur de séquence et/ou MLP sur descripteurs, puis tête commune.

    Les entrées sont passées dans l'ordre (séquence, descripteurs) en ne fournissant que celles
    qui existent : `model(x_seq, x_tab)`, `model(x_seq)` ou `model(x_tab)`.
    """

    def __init__(
        self,
        seq_encoder: SeqEncoder | None,
        tab_dim: int = 0,
        tab_hidden: int = 128,
        head_hidden: int = 128,
        out_dim: int = 5,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.seq = seq_encoder
        self.tab = (
            nn.Sequential(
                nn.Linear(tab_dim, tab_hidden),
                nn.BatchNorm1d(tab_hidden),
                make_activation(activation),
                nn.Dropout(dropout),
            )
            if tab_dim > 0
            else None
        )
        d = (seq_encoder.out_features if seq_encoder else 0) + (tab_hidden if tab_dim > 0 else 0)
        self.head = nn.Sequential(
            nn.Linear(d, head_hidden),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim),
        )

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        parts, i = [], 0
        if self.seq is not None:
            parts.append(self.seq(xs[i]))
            i += 1
        if self.tab is not None:
            parts.append(self.tab(xs[i]))
        return self.head(torch.cat(parts, dim=1))


def build_sequence_array(
    temperature: np.ndarray,
    excitation: np.ndarray,
    train_idx: np.ndarray,
    stats: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Construit (N, 3, L) = [T_int, ΔT_int, U/1000] standardisés avec les statistiques du train."""
    t = np.asarray(temperature, dtype=np.float32)
    x = np.stack([t, t - t[:, :1], np.asarray(excitation, dtype=np.float32) / 1000.0], axis=1)
    if stats is None:
        sub = x[train_idx]
        stats = {"mean": sub.mean(axis=(0, 2)), "std": sub.std(axis=(0, 2)) + 1e-6}
    x = (x - stats["mean"][None, :, None]) / stats["std"][None, :, None]
    return x.astype(np.float32), stats


# --------------------------------------------------------------------------------------
# Entraînement
# --------------------------------------------------------------------------------------
@dataclass
class TorchSettings:
    """Hyperparamètres d'entraînement communs à tous les réseaux."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    l1: float = 0.0
    batch_size: int = 512
    epochs: int = 60
    patience: int = 10
    optimizer: str = "adamw"  # adam, adamw, rmsprop, sgd
    scheduler: str = "plateau"  # none, plateau, cosine, onecycle
    loss: str = "mse"  # mse, mae, huber
    huber_delta: float = 1.0
    grad_clip: float = 1.0
    seed: int = 0
    input_noise: float = (
        0.0  # bruit gaussien ajouté aux entrées séquentielles (en écart-type standardisé)
    )


def _optimizer(params, s: TorchSettings) -> torch.optim.Optimizer:
    if s.optimizer == "adam":
        return torch.optim.Adam(params, lr=s.lr, weight_decay=s.weight_decay)
    if s.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=s.lr, weight_decay=s.weight_decay)
    if s.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=s.lr, weight_decay=s.weight_decay, momentum=0.9)
    if s.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=s.lr, weight_decay=s.weight_decay, momentum=0.9, nesterov=True
        )
    raise ValueError(s.optimizer)


def _loss(pred: torch.Tensor, target: torch.Tensor, s: TorchSettings) -> torch.Tensor:
    if s.loss == "mse":
        return F.mse_loss(pred, target)
    if s.loss == "mae":
        return F.l1_loss(pred, target)
    if s.loss == "huber":
        return F.huber_loss(pred, target, delta=s.huber_delta)
    raise ValueError(s.loss)


@torch.no_grad()
def predict_torch(
    model: nn.Module, inputs: tuple[torch.Tensor, ...], batch_size: int = 4096
) -> np.ndarray:
    """Prédictions z (N, 5) en mode évaluation."""
    model.eval()
    n = inputs[0].shape[0]
    outs = [
        model(*(x[i : i + batch_size] for x in inputs)).float().cpu()
        for i in range(0, n, batch_size)
    ]
    return torch.cat(outs).numpy()


def fit_torch(
    model: nn.Module,
    train_inputs: tuple[torch.Tensor, ...],
    z_train: torch.Tensor,
    val_inputs: tuple[torch.Tensor, ...],
    z_val: torch.Tensor,
    s: TorchSettings,
    trial=None,
    verbose: bool = False,
    seq_input_index: int | None = None,
) -> dict:
    """Entraîne `model` (déjà sur le bon device) avec arrêt anticipé sur la MSE de validation en z.

    Args:
        train_inputs / val_inputs: Tuples de tenseurs alignés (séquence et/ou descripteurs).
        trial: Essai Optuna optionnel (rapport par époque et élagage).
        seq_input_index: Indice de l'entrée séquentielle à bruiter si `s.input_noise > 0`.

    Returns:
        Dictionnaire : history (pertes), best_epoch, best_val_mse, seconds. Les meilleurs poids
        sont restaurés dans `model`.
    """
    torch.manual_seed(s.seed)
    np.random.seed(s.seed)
    opt = _optimizer(model.parameters(), s)
    n = z_train.shape[0]
    steps_per_epoch = math.ceil(n / s.batch_size)
    if s.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=max(2, s.patience // 3)
        )
    elif s.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s.epochs)
    elif s.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=s.lr, total_steps=s.epochs * steps_per_epoch
        )
    else:
        sched = None

    gen = torch.Generator(device="cpu").manual_seed(s.seed)
    best, best_state, bad, best_epoch = float("inf"), None, 0, -1
    history: dict[str, list[float]] = {"train": [], "val": []}
    start = time.perf_counter()
    for epoch in range(s.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(z_train.device)
        running = 0.0
        for k in range(steps_per_epoch):
            idx = perm[k * s.batch_size : (k + 1) * s.batch_size]
            if idx.numel() < 2:
                continue
            xb = [x[idx] for x in train_inputs]
            if s.input_noise > 0 and seq_input_index is not None:
                xb[seq_input_index] = xb[seq_input_index] + s.input_noise * torch.randn_like(
                    xb[seq_input_index]
                )
            pred = model(*xb)
            loss = _loss(pred, z_train[idx], s)
            if s.l1 > 0:
                loss = loss + s.l1 * sum(
                    p.abs().sum() for n_, p in model.named_parameters() if "weight" in n_
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if s.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), s.grad_clip)
            opt.step()
            if s.scheduler == "onecycle":
                sched.step()
            running += loss.detach().item() * idx.numel()

        val_pred = torch.from_numpy(predict_torch(model, val_inputs)).to(z_val.device)
        val_mse = F.mse_loss(val_pred, z_val).item()
        history["train"].append(running / n)
        history["val"].append(val_mse)
        if s.scheduler == "plateau":
            sched.step(val_mse)
        elif s.scheduler == "cosine":
            sched.step()
        if verbose:
            print(f"  époque {epoch + 1:3d} | train {running / n:.4f} | val MSE(z) {val_mse:.4f}")

        if val_mse < best - 1e-5:
            best, bad, best_epoch = val_mse, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if trial is not None:
            trial.report(val_mse, epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()
        if bad >= s.patience or not np.isfinite(val_mse):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_mse": best,
        "seconds": time.perf_counter() - start,
        "settings": asdict(s),
    }


# --------------------------------------------------------------------------------------
# Espaces de recherche Optuna
# --------------------------------------------------------------------------------------
def mlp_search_space(trial) -> tuple[dict, TorchSettings]:
    """Architecture + entraînement d'un MLP ou ResMLP tabulaire."""
    arch = trial.suggest_categorical("arch", ["mlp", "resmlp"])
    cfg = {
        "arch": arch,
        "activation": trial.suggest_categorical(
            "activation", ["relu", "gelu", "silu", "elu", "mish"]
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "norm": trial.suggest_categorical("norm", ["batch", "layer", "none"]),
        "init": trial.suggest_categorical("init", ["default", "kaiming", "xavier", "orthogonal"]),
    }
    if arch == "mlp":
        depth = trial.suggest_int("depth", 1, 5)
        width = trial.suggest_categorical("width", [64, 128, 256, 512, 1024])
        shape = trial.suggest_categorical("shape", ["constant", "funnel"])
        cfg["hidden"] = [
            max(32, width // (2**i)) if shape == "funnel" else width for i in range(depth)
        ]
    else:
        cfg["width"] = trial.suggest_categorical("width_res", [128, 256, 512])
        cfg["n_blocks"] = trial.suggest_int("n_blocks", 1, 6)
    s = TorchSettings(
        lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        l1=trial.suggest_categorical("l1", [0.0, 1e-7, 1e-6]),
        batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
        optimizer=trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop", "sgd"]),
        scheduler=trial.suggest_categorical("scheduler", ["plateau", "cosine", "onecycle", "none"]),
        loss=trial.suggest_categorical("loss", ["mse", "huber", "mae"]),
        epochs=60,
        patience=10,
    )
    if s.optimizer == "sgd":
        s.lr = trial.suggest_float("lr_sgd", 1e-3, 5e-2, log=True)
    return cfg, s


def build_tabular_net(cfg: dict, in_dim: int, out_dim: int = 5) -> nn.Module:
    """Instancie un MLP / ResMLP à partir d'une configuration (issue d'Optuna ou manuelle)."""
    if cfg["arch"] == "mlp":
        net = MLP(in_dim, out_dim, cfg["hidden"], cfg["activation"], cfg["dropout"], cfg["norm"])
    else:
        net = ResMLP(
            in_dim,
            out_dim,
            cfg["width"],
            cfg["n_blocks"],
            cfg["activation"],
            cfg["dropout"],
            cfg["norm"],
        )
    return init_weights(net, cfg.get("init", "default"))


def hybrid_search_space(trial) -> tuple[dict, TorchSettings]:
    """Architecture + entraînement du modèle hybride séquence + descripteurs."""
    cfg = {
        "kind": trial.suggest_categorical("kind", ["cnn_gru", "tcn", "cnn"]),
        "channels": trial.suggest_categorical("channels", [32, 64, 128]),
        "n_conv": trial.suggest_int("n_conv", 2, 4),
        "kernel": trial.suggest_categorical("kernel", [3, 5, 7]),
        "gru_hidden": trial.suggest_categorical("gru_hidden", [64, 128, 256]),
        "gru_layers": trial.suggest_int("gru_layers", 1, 2),
        "tab_hidden": trial.suggest_categorical("tab_hidden", [64, 128, 256]),
        "head_hidden": trial.suggest_categorical("head_hidden", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "use_tab": trial.suggest_categorical("use_tab", [True, False]),
    }
    s = TorchSettings(
        lr=trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024]),
        optimizer=trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        scheduler=trial.suggest_categorical("scheduler", ["plateau", "cosine", "onecycle"]),
        loss=trial.suggest_categorical("loss", ["mse", "huber"]),
        epochs=40,
        patience=8,
    )
    return cfg, s


def build_hybrid_net(cfg: dict, tab_dim: int, out_dim: int = 5) -> HybridNet:
    """Instancie un `HybridNet` à partir d'une configuration."""
    enc = SeqEncoder(
        cfg["kind"],
        3,
        cfg["channels"],
        cfg["n_conv"],
        cfg["kernel"],
        cfg["gru_hidden"],
        cfg["gru_layers"],
        cfg["dropout"],
    )
    return HybridNet(
        enc,
        tab_dim if cfg.get("use_tab", True) else 0,
        cfg["tab_hidden"],
        cfg["head_hidden"],
        out_dim,
        cfg["dropout"],
    )
