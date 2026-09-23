"""Feature engineering and the fitted design-matrix transformer.

`FeatureBuilder` is fit on train only (skew columns, one-hot vocabulary,
scaler) so nothing about the test distribution leaks into the model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import RobustScaler

QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLS = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
EXPOSURE_MAP = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
FINTYPE_MAP = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
GARAGE_FINISH_MAP = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
FUNCTIONAL_MAP = {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7}
FENCE_MAP = {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}
SLOPE_MAP = {"Gtl": 3, "Mod": 2, "Sev": 1}
SHAPE_MAP = {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1}
PAVED_MAP = {"Y": 2, "P": 1, "N": 0}


def _ordinal(X: pd.DataFrame, col: str, mapping: dict[str, int]) -> None:
    if col in X:
        X[col] = X[col].map(mapping).fillna(0).astype(int)


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Add domain features and encode ordinal quality scales. Stateless."""
    X = X.copy()

    for col in QUALITY_COLS:
        _ordinal(X, col, QUALITY_MAP)
    _ordinal(X, "BsmtExposure", EXPOSURE_MAP)
    _ordinal(X, "BsmtFinType1", FINTYPE_MAP)
    _ordinal(X, "BsmtFinType2", FINTYPE_MAP)
    _ordinal(X, "GarageFinish", GARAGE_FINISH_MAP)
    _ordinal(X, "Functional", FUNCTIONAL_MAP)
    _ordinal(X, "Fence", FENCE_MAP)
    _ordinal(X, "LandSlope", SLOPE_MAP)
    _ordinal(X, "LotShape", SHAPE_MAP)
    _ordinal(X, "PavedDrive", PAVED_MAP)

    # Size aggregates — the single strongest family of features on Ames.
    X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
    X["TotalFinSF"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["1stFlrSF"] + X["2ndFlrSF"]
    X["TotalPorchSF"] = (X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"]
                        + X["ScreenPorch"] + X["WoodDeckSF"])
    X["TotalBath"] = (X["FullBath"] + 0.5 * X["HalfBath"]
                      + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"])

    # Age and renovation.
    yr_sold = X["YrSold"].astype(int)
    X["HouseAge"] = yr_sold - X["YearBuilt"]
    X["RemodAge"] = yr_sold - X["YearRemodAdd"]
    X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
    X["IsNew"] = (yr_sold == X["YearBuilt"]).astype(int)
    X["GarageAge"] = np.where(X["GarageYrBlt"] > 0, yr_sold - X["GarageYrBlt"], 0)

    # Presence flags for sparse amenities.
    for src, flag in [("PoolArea", "HasPool"), ("2ndFlrSF", "Has2ndFloor"),
                      ("GarageArea", "HasGarage"), ("TotalBsmtSF", "HasBsmt"),
                      ("Fireplaces", "HasFireplace")]:
        X[flag] = (X[src] > 0).astype(int)

    # Interaction of overall quality with size — quality is priced per sqft.
    X["QualxSF"] = X["OverallQual"] * X["TotalSF"]
    X["QualxGrLiv"] = X["OverallQual"] * X["GrLivArea"]
    X["OverallScore"] = X["OverallQual"] * X["OverallCond"]

    # Very sparse / redundant raw columns.
    X = X.drop(columns=[c for c in ["Street", "PoolQC"] if c in X])
    return X


class FeatureBuilder:
    """Fit-on-train transformer: skew correction, one-hot, robust scaling."""

    def __init__(self, skew_threshold: float = 0.75) -> None:
        self.skew_threshold = skew_threshold
        self.skewed_cols_: list[str] = []
        self.columns_: list[str] = []
        self.scaler_ = RobustScaler()
        # Learned imputation statistics for LotFrontage (train only).
        self.frontage_by_nbhd_: pd.Series = pd.Series(dtype=float)
        self.frontage_global_: float = 0.0

    def fit(self, X: pd.DataFrame) -> "FeatureBuilder":
        if "LotFrontage" in X:
            self.frontage_by_nbhd_ = X.groupby("Neighborhood")["LotFrontage"].median()
            self.frontage_global_ = float(X["LotFrontage"].median())
        X = self._impute_frontage(X)
        num = X.select_dtypes(include=[np.number])
        skews = num.apply(lambda s: skew(s.dropna()))
        # Do not log-transform 0/1 flags or ordinal scores.
        candidates = [c for c in num.columns if num[c].nunique() > 10]
        self.skewed_cols_ = [c for c in candidates if abs(skews[c]) > self.skew_threshold]
        X = self._log_skewed(X)
        X = pd.get_dummies(X, dtype=float)
        self.columns_ = X.columns.tolist()
        self.scaler_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._impute_frontage(X)
        X = self._log_skewed(X)
        X = pd.get_dummies(X, dtype=float)
        X = X.reindex(columns=self.columns_, fill_value=0.0)
        return pd.DataFrame(self.scaler_.transform(X), columns=self.columns_, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def _impute_frontage(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill LotFrontage with the *train* neighbourhood median, then train global."""
        if "LotFrontage" not in X:
            return X
        X = X.copy()
        fill = X["Neighborhood"].map(self.frontage_by_nbhd_)
        X["LotFrontage"] = X["LotFrontage"].fillna(fill).fillna(self.frontage_global_)
        return X

    def _log_skewed(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.skewed_cols_:
            if c in X:
                X[c] = np.log1p(X[c].clip(lower=0))
        return X


def build_design(X_train: pd.DataFrame, X_test: pd.DataFrame | None, cfg: dict[str, Any]):
    """Convenience: clean+engineer both frames, fit builder on train, transform both."""
    from .preprocessing import clean  # local import to keep module graph acyclic

    builder = FeatureBuilder(skew_threshold=cfg["preprocessing"]["skew_threshold"])
    Xt = builder.fit_transform(engineer(clean(X_train)))
    Xs = builder.transform(engineer(clean(X_test))) if X_test is not None else None
    return Xt, Xs, builder
