"""Cleaning: outliers, missing-value semantics, dtype corrections.

Every rule here is derived from the Ames data dictionary, not from the
target — so the same function is safe to run on train and test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# In these columns NA literally means "the house does not have this".
NONE_MEANS_ABSENT = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

# Numeric companions of the above: absent feature -> zero quantity.
ZERO_MEANS_ABSENT = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

# Genuinely missing categoricals: fill with the mode.
MODE_FILL = ["MSZoning", "Electrical", "KitchenQual", "Exterior1st",
             "Exterior2nd", "SaleType", "Functional"]

# Numeric codes that are really categories.
CATEGORICAL_NUMERICS = ["MSSubClass", "MoSold", "YrSold"]


def drop_outliers(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Remove the documented Ames outliers (huge living area, low price).

    Training-only. De Cock's paper recommends removing these; they are
    partial sales that no model should be asked to fit.
    """
    rule = cfg["preprocessing"]["drop_outliers"]
    mask = (X[rule["column"]] > rule["threshold"]) & (y < rule["price_below"])
    return X.loc[~mask].reset_index(drop=True), y.loc[~mask].reset_index(drop=True)


def clean(X: pd.DataFrame) -> pd.DataFrame:
    """Apply missing-value semantics and dtype fixes. Stateless."""
    X = X.copy()

    for col in NONE_MEANS_ABSENT:
        if col in X:
            X[col] = X[col].fillna("None")
    for col in ZERO_MEANS_ABSENT:
        if col in X:
            X[col] = X[col].fillna(0)

    # Utilities is constant on the Kaggle data (all but one "AllPub"); it
    # carries no signal and breaks one-hot alignment between train/test.
    X = X.drop(columns=[c for c in ["Utilities"] if c in X])

    # LotFrontage is deliberately left NaN here: its neighbourhood-median fill
    # is a *learned* statistic, so FeatureBuilder fits it on train and applies
    # it to test (see features.py). Everything else in this function is a rule.

    for col in MODE_FILL:
        if col in X:
            X[col] = X[col].fillna(X[col].mode().iloc[0])

    # Anything numeric still missing -> 0 (only a handful of cells remain),
    # except LotFrontage, which FeatureBuilder imputes from train statistics.
    num_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c != "LotFrontage"]
    X[num_cols] = X[num_cols].fillna(0)

    # Numeric codes -> categorical strings.
    for col in CATEGORICAL_NUMERICS:
        if col in X:
            X[col] = X[col].astype(int).astype(str)

    # A garage built in 2207 is a typo for 2007.
    if "GarageYrBlt" in X:
        X.loc[X["GarageYrBlt"] > 2020, "GarageYrBlt"] = 2007

    return X
