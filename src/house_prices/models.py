"""Model zoo and the out-of-fold blender.

All models are fit on log1p(SalePrice); predictions are in log space until
`expm1` at the very end. Weights are learned on OOF predictions so the blend
never sees a model's in-sample fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor


def make_models(cfg: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """Instantiate every model named in config with its hyper-parameters."""
    p = cfg["models"]
    return {
        "lasso": Lasso(random_state=seed, **p["lasso"]),
        "elasticnet": ElasticNet(random_state=seed, **p["elasticnet"]),
        "ridge": Ridge(**p["ridge"]),
        "svr": SVR(**p["svr"]),
        "gbr": GradientBoostingRegressor(random_state=seed, **p["gbr"]),
        "lightgbm": LGBMRegressor(random_state=seed, **p["lightgbm"]),
        "xgboost": XGBRegressor(random_state=seed, **p["xgboost"]),
        "catboost": CatBoostRegressor(random_seed=seed, **p["catboost"]),
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def optimise_weights(oof: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Non-negative weights summing to 1 that minimise OOF RMSE."""
    k = oof.shape[1]
    x0 = np.full(k, 1.0 / k)

    def loss(w: np.ndarray) -> float:
        return rmse(y, oof.values @ w)

    res = minimize(loss, x0, method="SLSQP", bounds=[(0, 1)] * k,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    return pd.Series(w, index=oof.columns)


class Blend:
    """Weighted average of fitted base models, in log space."""

    def __init__(self, models: dict[str, Any], weights: pd.Series) -> None:
        self.models = models
        self.weights = weights

    def fit(self, X: pd.DataFrame, y_log: np.ndarray) -> "Blend":
        for name, m in self.models.items():
            if self.weights.get(name, 0) > 0:
                m.fit(X, y_log)
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(X))
        for name, m in self.models.items():
            w = self.weights.get(name, 0)
            if w > 0:
                out += w * m.predict(X)
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.expm1(self.predict_log(X))
