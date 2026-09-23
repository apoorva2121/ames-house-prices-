"""Smoke tests: the pipeline runs end to end and train/test designs align."""

import numpy as np
import pandas as pd

from house_prices.data import load_config, load_train
from house_prices.features import build_design
from house_prices.preprocessing import clean, drop_outliers


def test_clean_leaves_only_lotfrontage_missing():
    """Rules fill everything except LotFrontage, which is a learned imputation."""
    cfg = load_config()
    X, _, _ = load_train(cfg)
    missing = clean(X).isna().sum()
    assert set(missing[missing > 0].index) <= {"LotFrontage"}


def test_design_matrix_has_no_missing():
    cfg = load_config()
    X, _, _ = load_train(cfg)
    Xt, _, builder = build_design(X, None, cfg)
    assert not np.isnan(Xt.values).any()
    assert builder.frontage_global_ > 0


def test_outlier_rule_drops_two_rows():
    cfg = load_config()
    X, y, _ = load_train(cfg)
    X2, y2 = drop_outliers(X, y, cfg)
    assert len(X) - len(X2) == 2
    assert len(X2) == len(y2)


def test_train_test_design_align():
    cfg = load_config()
    X, y, _ = load_train(cfg)
    # simulate a test frame with an unseen category and a missing column
    X_test = X.sample(50, random_state=0).copy()
    X_test.loc[X_test.index[0], "Neighborhood"] = "Nowhere"
    Xt, Xs, _ = build_design(X, X_test, cfg)
    assert list(Xt.columns) == list(Xs.columns)
    assert not np.isnan(Xs.values).any()
