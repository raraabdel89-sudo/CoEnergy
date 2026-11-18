"""Orchestration des expériences : un `ExperimentContext` centralise les données, le découpage,
les descripteurs, le journal et les prédictions, et expose une méthode par famille de modèles.

Règles garanties par construction :
- toutes les statistiques (prétraitement, cibles, séquences) sont ajustées sur le **train** ;
- les choix (features, hyperparamètres, poids d'ensemble) se font sur la **validation** ;
- les prédictions de **test** sont stockées mais jamais utilisées avant l'évaluation finale.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from coenergy.evaluation import ExperimentLog, Timer, indicator_metrics
from coenergy.fe import TabularPreprocessor
from coenergy.predict import Member
from coenergy.targets import TargetTransform


def r2_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² moyen sur les 6 indicateurs (unités physiques)."""
    return float(
        indicator_metrics(y_true, y_pred, [str(i) for i in range(y_true.shape[1])])["r2"].mean()
    )


@dataclass
class ExperimentContext:
    """État partagé des expériences."""

    y: np.ndarray  # indicateurs physiques (N, 6)
    labels: list[str]
    splits: dict[str, np.ndarray]
    groups: np.ndarray
    feats: pd.DataFrame
    temperature: np.ndarray  # (N, L), pas `dt_minutes`
    excitation: np.ndarray
    dt_minutes: float
    log: ExperimentLog
    out_dir: Path
    device: str = "cpu"
    seed: int = 42
    train_eval_size: int = 20_000
    preds_train: dict[str, np.ndarray] = field(default_factory=dict)
    preds_val: dict[str, np.ndarray] = field(default_factory=dict)
    preds_test: dict[str, np.ndarray] = field(default_factory=dict)
    members: dict[str, Member] = field(default_factory=dict)
    configs: dict[str, dict] = field(default_factory=dict)
    _prep_cache: dict = field(default_factory=dict)
    _tensor_cache: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        tr = self.splits["train"]
        self.train_eval = np.sort(
            rng.choice(tr, size=min(self.train_eval_size, len(tr)), replace=False)
        )
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ utilitaires
    @property
    def eval_splits(self) -> dict[str, np.ndarray]:
        """Sous-ensembles évalués : extrait du train (pour l'écart train/val), val, test."""
        return {
            "train_eval": self.train_eval,
            "val": self.splits["val"],
            "test": self.splits["test"],
        }

    def prepared(
        self, cols: list[str], scaling: str = "standard"
    ) -> tuple[TabularPreprocessor, dict]:
        """Prétraitement ajusté sur le train, et matrices pour chaque sous-ensemble (en cache)."""
        key = (tuple(cols), scaling)
        if key not in self._prep_cache:
            pp = TabularPreprocessor(scaling).fit(self.feats.iloc[self.splits["train"]][cols])
            idx = {"train": self.splits["train"], **self.eval_splits}
            x = {k: pp.transform(self.feats.iloc[v][cols]) for k, v in idx.items()}
            self._prep_cache[key] = (pp, x)
        return self._prep_cache[key]

    def target(self, param: str) -> TargetTransform:
        """Transformation des cibles ajustée sur le train."""
        return TargetTransform(param).fit(self.y[self.splits["train"]])

    def record(
        self,
        name: str,
        family: str,
        params: dict,
        preds: dict[str, np.ndarray],
        seconds: float,
        member: Member | None = None,
        **extra: object,
    ) -> dict:
        """Journalise une expérience (train/val) et stocke ses prédictions val et test."""
        row = self.log.add(
            name,
            family,
            params,
            self.y[self.train_eval],
            preds["train_eval"],
            self.y[self.splits["val"]],
            preds["val"],
            seconds,
            **extra,
        )
        self.preds_train[name] = preds["train_eval"]
        self.preds_val[name] = preds["val"]
        self.preds_test[name] = preds["test"]
        if member is not None:
            self.members[name] = member
        np.savez_compressed(
            self.out_dir / "predictions.npz",
            **{f"val__{k}": v for k, v in self.preds_val.items()},
            **{f"test__{k}": v for k, v in self.preds_test.items()},
        )
        return row

    # ----------------------------------------------------------- modèles tabulaires classiques
    def run_sklearn(
        self,
        name: str,
        family: str,
        fit_fn: Callable,
        cols: list[str],
        param: str = "A",
        scaling: str = "standard",
        params: dict | None = None,
        log: bool = True,
    ):
        """Entraîne un modèle « sklearn-like » (`fit_fn(Xtr, Ztr, Xva, Zva)` → MultiTargetModel)."""
        pp, x = self.prepared(cols, scaling)
        tt = self.target(param)
        z_tr = tt.encode(self.y[self.splits["train"]])
        z_va = tt.encode(self.y[self.splits["val"]])
        with Timer() as t:
            model = fit_fn(x["train"], z_tr, x["val"], z_va)
        preds = {k: tt.decode(model.predict(x[k])) for k in self.eval_splits}
        member = Member(name, "sklearn", tt, list(cols), pp, model)
        info = {"param": param, "scaling": scaling, "n_features": len(cols), **(params or {})}
        if getattr(model, "best_iterations", None):
            info["best_iterations"] = model.best_iterations
        if not log:
            return model, preds, member
        self.configs[name] = {
            "type": "sklearn",
            "family": family,
            "fit_fn": fit_fn,
            "cols": list(cols),
            "param": param,
            "scaling": scaling,
            "params": params or {},
        }
        row = self.record(name, family, info, preds, t.seconds, member)
        return model, row

    # ------------------------------------------------------------------ réseaux PyTorch
    def _tab_tensors(self, cols: list[str], scaling: str):
        import torch

        key = ("tab", tuple(cols), scaling)
        if key not in self._tensor_cache:
            _, x = self.prepared(cols, scaling)
            self._tensor_cache[key] = {k: torch.from_numpy(v).to(self.device) for k, v in x.items()}
        return self._tensor_cache[key]

    def _seq_tensors(self):
        import torch

        from coenergy.nets import build_sequence_array

        if "seq" not in self._tensor_cache:
            arr, stats = build_sequence_array(
                self.temperature, self.excitation, self.splits["train"]
            )
            idx = {"train": self.splits["train"], **self.eval_splits}
            self._tensor_cache["seq"] = (
                {k: torch.from_numpy(arr[v]).to(self.device) for k, v in idx.items()},
                stats,
            )
        return self._tensor_cache["seq"]

    def fit_net(
        self,
        kind: str,
        cfg: dict,
        settings,
        cols: list[str],
        param: str = "A",
        scaling: str = "standard",
        trial=None,
        verbose: bool = False,
    ):
        """Entraîne un réseau ("tabular_nn" ou "hybrid"). Renvoie (net, preds, member, info)."""
        import torch

        from coenergy.nets import build_hybrid_net, build_tabular_net, fit_torch, predict_torch

        torch.manual_seed(settings.seed)
        tt = self.target(param)
        z = {
            k: torch.from_numpy(tt.encode(self.y[v])).to(self.device)
            for k, v in {"train": self.splits["train"], "val": self.splits["val"]}.items()
        }
        pp = None
        if kind == "tabular_nn":
            pp, _ = self.prepared(cols, scaling)
            tab = self._tab_tensors(cols, scaling)
            net = build_tabular_net(cfg, tab["train"].shape[1]).to(self.device)
            inputs = {k: (tab[k],) for k in tab}
            tab_dim, seq_stats, seq_index = tab["train"].shape[1], None, None
        else:
            seq, seq_stats = self._seq_tensors()
            use_tab = cfg.get("use_tab", True) and len(cols) > 0
            cfg = {**cfg, "use_tab": use_tab}
            tab_dim = 0
            inputs = {k: (seq[k],) for k in seq}
            if use_tab:
                pp, _ = self.prepared(cols, scaling)
                tab = self._tab_tensors(cols, scaling)
                tab_dim = tab["train"].shape[1]
                inputs = {k: (seq[k], tab[k]) for k in seq}
            net = build_hybrid_net(cfg, tab_dim).to(self.device)
            seq_index = 0
        info = fit_torch(
            net,
            inputs["train"],
            z["train"],
            inputs["val"],
            z["val"],
            settings,
            trial=trial,
            verbose=verbose,
            seq_input_index=seq_index,
        )
        preds = {k: tt.decode(predict_torch(net, inputs[k])) for k in self.eval_splits}
        state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        member = Member(
            "",
            kind,
            tt,
            list(cols) if pp is not None else [],
            pp,
            None,
            cfg,
            state,
            tab_dim,
            seq_stats,
        )
        return net, preds, member, info

    def run_net(
        self,
        name: str,
        family: str,
        kind: str,
        cfg: dict,
        settings,
        cols: list[str],
        param: str = "A",
        scaling: str = "standard",
        verbose: bool = False,
    ) -> dict:
        """Entraîne un réseau et journalise le résultat."""
        from dataclasses import asdict

        net, preds, member, info = self.fit_net(
            kind, cfg, settings, cols, param, scaling, verbose=verbose
        )
        member.name = name
        self.configs[name] = {
            "type": "net",
            "family": family,
            "kind": kind,
            "cfg": cfg,
            "settings": settings,
            "cols": list(cols),
            "param": param,
            "scaling": scaling,
        }
        params = {
            "param": param,
            "scaling": scaling,
            "n_features": len(cols),
            **cfg,
            **asdict(settings),
            "best_epoch": info["best_epoch"] + 1,
        }
        return self.record(
            name, family, params, preds, info["seconds"], member, best_epoch=info["best_epoch"] + 1
        )

    # ------------------------------------------------------------------ évaluation finale
    def test_table(self, names: list[str]) -> pd.DataFrame:
        """Métriques de test (évaluation finale uniquement)."""
        y_te = self.y[self.splits["test"]]
        rows = {}
        for n in names:
            m = indicator_metrics(y_te, self.preds_test[n], self.labels)
            rows[n] = {
                "r2_test_mean": m["r2"].mean(),
                **{f"r2_test_{lab}": m.loc[lab, "r2"] for lab in self.labels},
                "rmse_test_HLC": m.loc[self.labels[0], "rmse"],
                "mae_test_HLC": m.loc[self.labels[0], "mae"],
                "p95_test_HLC": m.loc[self.labels[0], "p95_abs"],
            }
        return pd.DataFrame(rows).T.sort_values("r2_test_mean", ascending=False)
