"""Modèles non neuronaux de comparaison : boosting (LightGBM, XGBoost, CatBoost), forêts,
SVR et ridge. Chaque modèle prédit les 5 composantes de z (voir `targets.py`), un modèle par
composante pour le boosting et la SVR, un modèle multi-sorties pour les forêts et la ridge.

Les bibliothèques de boosting sont importées à la demande : ce module reste importable sans
elles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MultiTargetModel:
    """Regroupe un modèle par composante de z (ou un seul modèle multi-sorties)."""

    kind: str
    models: list = field(default_factory=list)
    multioutput: bool = False
    best_iterations: list[int] = field(default_factory=list)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Prédit z, forme (N, 5)."""
        if self.multioutput:
            return np.asarray(self.models[0].predict(x), dtype=np.float32)
        return np.column_stack([m.predict(x) for m in self.models]).astype(np.float32)


# --------------------------------------------------------------------------------------
# Boosting
# --------------------------------------------------------------------------------------
LGBM_DEFAULTS = dict(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    max_bin=255,
    verbose=-1,
)


def fit_lightgbm(
    x_tr, z_tr, x_va, z_va, params: dict | None = None, seed: int = 0, early_stopping: int = 150
) -> MultiTargetModel:
    """LightGBM, un booster par composante, arrêt anticipé sur la validation."""
    import lightgbm as lgb

    p = {**LGBM_DEFAULTS, **(params or {}), "random_state": seed, "n_jobs": -1}
    out = MultiTargetModel("lightgbm")
    for j in range(z_tr.shape[1]):
        m = lgb.LGBMRegressor(**p)
        m.fit(
            x_tr,
            z_tr[:, j],
            eval_set=[(x_va, z_va[:, j])],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        out.models.append(m)
        out.best_iterations.append(int(m.best_iteration_ or p["n_estimators"]))
    return out


def lgbm_search_space(trial) -> dict:
    """Espace de recherche Optuna pour LightGBM."""
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 511, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 500, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
        max_bin=trial.suggest_categorical("max_bin", [63, 127, 255]),
    )


XGB_DEFAULTS = dict(
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
)


def fit_xgboost(
    x_tr,
    z_tr,
    x_va,
    z_va,
    params: dict | None = None,
    seed: int = 0,
    early_stopping: int = 150,
    device: str = "cuda",
) -> MultiTargetModel:
    """XGBoost (GPU si disponible), un modèle par composante."""
    import xgboost as xgb

    p = {
        **XGB_DEFAULTS,
        **(params or {}),
        "random_state": seed,
        "device": device,
        "early_stopping_rounds": early_stopping,
    }
    out = MultiTargetModel("xgboost")
    for j in range(z_tr.shape[1]):
        m = xgb.XGBRegressor(**p)
        m.fit(x_tr, z_tr[:, j], eval_set=[(x_va, z_va[:, j])], verbose=False)
        out.models.append(m)
        out.best_iterations.append(int(m.best_iteration))
    return out


CAT_DEFAULTS = dict(
    iterations=5000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    loss_function="RMSE",
    verbose=False,
)


def fit_catboost(
    x_tr,
    z_tr,
    x_va,
    z_va,
    params: dict | None = None,
    seed: int = 0,
    early_stopping: int = 150,
    task_type: str = "GPU",
) -> MultiTargetModel:
    """CatBoost (GPU si disponible), un modèle par composante."""
    from catboost import CatBoostRegressor

    p = {
        **CAT_DEFAULTS,
        **(params or {}),
        "random_seed": seed,
        "task_type": task_type,
        "od_type": "Iter",
        "od_wait": early_stopping,
    }
    out = MultiTargetModel("catboost")
    for j in range(z_tr.shape[1]):
        m = CatBoostRegressor(**p)
        m.fit(x_tr, z_tr[:, j], eval_set=(x_va, z_va[:, j]), use_best_model=True)
        out.models.append(m)
        out.best_iterations.append(int(m.get_best_iteration()))
    return out


# --------------------------------------------------------------------------------------
# scikit-learn
# --------------------------------------------------------------------------------------
def fit_forest(
    kind: str, x_tr, z_tr, params: dict | None = None, seed: int = 0
) -> MultiTargetModel:
    """Random Forest ("rf") ou Extra Trees ("et"), multi-sorties natif."""
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    defaults = dict(
        n_estimators=300, min_samples_leaf=5, max_features=0.5, max_samples=0.5, n_jobs=-1
    )
    p = {**defaults, **(params or {}), "random_state": seed}
    cls = RandomForestRegressor if kind == "rf" else ExtraTreesRegressor
    if cls is ExtraTreesRegressor:
        p["bootstrap"] = True
    model = cls(**p).fit(x_tr, z_tr)
    return MultiTargetModel(kind, [model], multioutput=True)


def fit_svr(
    x_tr, z_tr, params: dict | None = None, seed: int = 0, n_sub: int = 15_000
) -> MultiTargetModel:
    """SVR à noyau RBF, un modèle par composante, sur un sous-échantillon (coût en O(n²))."""
    from sklearn.svm import SVR

    p = {"C": 3.0, "epsilon": 0.05, "gamma": "scale", **(params or {})}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_tr), size=min(n_sub, len(x_tr)), replace=False)
    out = MultiTargetModel("svr")
    for j in range(z_tr.shape[1]):
        out.models.append(SVR(**p).fit(x_tr[idx], z_tr[idx, j]))
    return out


def fit_ridge(x_tr, z_tr, alpha: float = 1.0) -> MultiTargetModel:
    """Régression ridge multi-sorties."""
    from sklearn.linear_model import Ridge

    return MultiTargetModel("ridge", [Ridge(alpha=alpha).fit(x_tr, z_tr)], multioutput=True)


def fit_knn(x_tr, z_tr, n_neighbors: int = 50) -> MultiTargetModel:
    """k plus proches voisins (pondération par la distance)."""
    from sklearn.neighbors import KNeighborsRegressor

    m = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance", n_jobs=-1).fit(x_tr, z_tr)
    return MultiTargetModel("knn", [m], multioutput=True)


def feature_importance(model: MultiTargetModel, names: list[str]) -> np.ndarray:
    """Importance moyenne (gain normalisé) sur les composantes, pour les modèles à arbres."""
    imps = []
    for m in model.models:
        if hasattr(m, "booster_"):
            imp = m.booster_.feature_importance(importance_type="gain").astype(float)
        elif hasattr(m, "get_booster"):
            score = m.get_booster().get_score(importance_type="gain")
            imp = np.array([score.get(f"f{i}", 0.0) for i in range(len(names))])
        elif hasattr(m, "get_feature_importance"):
            imp = np.asarray(m.get_feature_importance(), dtype=float)
        elif hasattr(m, "feature_importances_"):
            imp = np.asarray(m.feature_importances_, dtype=float)
        else:
            continue
        imps.append(imp / max(imp.sum(), 1e-12))
    return np.mean(imps, axis=0) if imps else np.zeros(len(names))


def xgb_search_space(trial) -> dict:
    """Espace de recherche Optuna pour XGBoost (rapide sur GPU)."""
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 200.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
        max_bin=trial.suggest_categorical("max_bin", [64, 128, 256]),
    )
