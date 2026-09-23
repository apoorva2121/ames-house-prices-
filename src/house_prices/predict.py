"""Predict: load artifacts, transform Kaggle test.csv, write submission.csv.

Usage:
    python -m house_prices.predict [--config config.yaml] [--out submission.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import load_config, load_test
from .features import engineer
from .preprocessing import clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # joblib uses pickle; only load artifacts produced by our own train.py.
    # Never point this at a model file from an untrusted source.
    bundle = joblib.load(Path(cfg["paths"]["artifacts"]) / "model.joblib")
    builder, blend = bundle["builder"], bundle["blend"]

    X_test, ids = load_test(cfg)
    Xd = builder.transform(engineer(clean(X_test)))
    preds = blend.predict(Xd)

    sub = pd.DataFrame({cfg["id_column"]: ids, cfg["target"]: preds})
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(sub)} rows, "
          f"price range {preds.min():,.0f} - {preds.max():,.0f}")


if __name__ == "__main__":
    main()
