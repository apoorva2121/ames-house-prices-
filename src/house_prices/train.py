"""Train: evaluate the zoo, learn blend weights, fit the final blend, save artifacts.

Usage:
    python -m house_prices.train [--config config.yaml] [--fast]

`--fast` shrinks n_repeats to 1 and boosting rounds to 600 for a quick check.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import load_config, load_train
from .evaluate import cross_validate
from .features import build_design
from .models import Blend, make_models, optimise_weights, rmse
from .preprocessing import drop_outliers

# Kaggle public leaderboard references (RMSE on log price), as of Sep 2026.
# Source: competition page + the "solution file" dataset writeup by Carl McBride Ellis.
BENCHMARKS = {
    "kaggle_public_lb_median": 0.13584,
    "typical_strong_notebook": 0.120,
    "legitimate_top_scores": 0.113,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fast:
        cfg["cv"]["n_repeats"] = 1
        for k in ("gbr", "lightgbm", "xgboost"):
            cfg["models"][k]["n_estimators"] = 600
        cfg["models"]["catboost"]["iterations"] = 600

    art = Path(cfg["paths"]["artifacts"]); art.mkdir(parents=True, exist_ok=True)
    rep = Path(cfg["paths"]["reports"]); rep.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Data
    X, y, _ = load_train(cfg)
    X, y = drop_outliers(X, y, cfg)
    y_log = np.log1p(y.values)
    Xd, _, builder = build_design(X, None, cfg)
    print(f"design matrix: {Xd.shape[0]} rows x {Xd.shape[1]} features "
          f"({len(builder.skewed_cols_)} log-transformed)")

    # 2. Cross-validate every model
    oof, scores = cross_validate(lambda seed: make_models(cfg, seed), Xd, y_log, cfg)
    print("\nPer-model out-of-fold RMSLE:\n", scores.round(5).to_string())

    # 3. Blend weights on OOF
    if cfg["blend"]["fixed"]:
        weights = pd.Series(cfg["blend"]["fixed"])
    else:
        weights = optimise_weights(oof, y_log)
    blend_oof = rmse(y_log, oof[weights.index].values @ weights.values)
    print("\nBlend weights:\n", weights.round(4).to_string())
    print(f"\nBLEND out-of-fold RMSLE: {blend_oof:.5f}")
    for k, v in BENCHMARKS.items():
        print(f"  vs {k:<28} {v:.5f}  ->  {'better' if blend_oof < v else 'worse'} by {abs(v - blend_oof):.5f}")

    # 4. Fit final blend on all training data and persist
    final = Blend(make_models(cfg, cfg["cv"]["random_state"]), weights).fit(Xd, y_log)
    joblib.dump({"builder": builder, "blend": final, "config": cfg}, art / "model.joblib")

    report = {
        "n_train_rows": int(Xd.shape[0]),
        "n_features": int(Xd.shape[1]),
        "cv": cfg["cv"],
        "per_model": scores.round(5).to_dict(),
        "blend_weights": weights.round(4).to_dict(),
        "blend_oof_rmsle": round(blend_oof, 5),
        "benchmarks": BENCHMARKS,
        "runtime_seconds": round(time.time() - t0, 1),
        "fast_mode": args.fast,
    }
    (rep / "cv_report.json").write_text(json.dumps(report, indent=2))
    oof.assign(y_log=y_log).to_csv(rep / "oof_predictions.csv", index=False)
    print(f"\nartifacts -> {art / 'model.joblib'}\nreport    -> {rep / 'cv_report.json'}"
          f"\nruntime   -> {report['runtime_seconds']}s")


if __name__ == "__main__":
    main()
