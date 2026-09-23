"""Repeated K-fold evaluation producing out-of-fold predictions per model.

The metric is RMSE on log1p(SalePrice) — exactly Kaggle's leaderboard
metric for this competition (commonly written RMSLE).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from .models import rmse


def cross_validate(
    model_factory: Callable[[int], dict[str, Any]],
    X: pd.DataFrame,
    y_log: np.ndarray,
    cfg: dict[str, Any],
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (oof_predictions, per_model_scores).

    OOF predictions are averaged over repeats so every row gets one value
    per model; scores report mean ± std across all folds.
    """
    cv = cfg["cv"]
    rkf = RepeatedKFold(n_splits=cv["n_splits"], n_repeats=cv["n_repeats"],
                        random_state=cv["random_state"])
    names = list(model_factory(0).keys())
    oof_sum = pd.DataFrame(0.0, index=X.index, columns=names)
    oof_cnt = pd.Series(0, index=X.index)
    fold_scores: dict[str, list[float]] = {n: [] for n in names}

    for fold, (tr, va) in enumerate(rkf.split(X)):
        models = model_factory(cv["random_state"] + fold)
        for name, m in models.items():
            m.fit(X.iloc[tr], y_log[tr])
            pred = m.predict(X.iloc[va])
            oof_sum.iloc[va, oof_sum.columns.get_loc(name)] += pred
            fold_scores[name].append(rmse(y_log[va], pred))
        oof_cnt.iloc[va] += 1
        log(f"fold {fold + 1:02d}/{cv['n_splits'] * cv['n_repeats']}  "
            + "  ".join(f"{n}={fold_scores[n][-1]:.4f}" for n in names))

    oof = oof_sum.div(oof_cnt, axis=0)
    scores = pd.DataFrame({
        "cv_rmsle_mean": {n: np.mean(v) for n, v in fold_scores.items()},
        "cv_rmsle_std": {n: np.std(v) for n, v in fold_scores.items()},
        "oof_rmsle": {n: rmse(y_log, oof[n].values) for n in names},
    }).sort_values("oof_rmsle")
    return oof, scores
