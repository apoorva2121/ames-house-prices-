"""Data loading and configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Read the YAML config into a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_train(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (features, target, ids) for the training set.

    The target is returned untransformed; callers decide on log1p.
    """
    df = pd.read_csv(cfg["paths"]["train"])
    target, idc = cfg["target"], cfg["id_column"]
    y = df[target].astype(float)
    ids = df[idc]
    X = df.drop(columns=[target, idc])
    return X, y, ids


def load_test(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features, ids) for the Kaggle test set."""
    path = Path(cfg["paths"]["test"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download test.csv from the Kaggle competition page "
            "and place it there to generate a submission."
        )
    df = pd.read_csv(path)
    idc = cfg["id_column"]
    return df.drop(columns=[idc]), df[idc]
