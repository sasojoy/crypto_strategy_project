"""Workflow-oriented training script for the h16 strategy.

This module provides a CLI entrypoint as well as reusable helpers that are
consumed by the CI orchestrator.  It implements a time-series aware training
loop, probability calibration and threshold scanning so that downstream jobs can
reason about expected performance without re-running the training code.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from csp.features.h16 import make_labels

DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "max_depth": 3,
    "n_estimators": 400,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": 4,
}


@dataclass
class TrainingDataset:
    """Container for the prepared training frame."""

    features: pd.DataFrame
    labels: pd.Series
    future_returns: pd.Series
    feature_columns: List[str]
    timestamps: pd.Series
    metadata: Dict[str, Any]


def load_feature_builder(path: str):
    """Dynamically load the feature builder function from a dotted path."""

    module_name, func_name = path.rsplit(".", 1)
    module = import_module(module_name)
    feature_func = getattr(module, func_name)
    return feature_func


def compute_future_returns(frame: pd.DataFrame, horizon: int) -> pd.Series:
    future_close = frame["close"].shift(-horizon)
    returns = (future_close - frame["close"]) / frame["close"]
    return returns


def prepare_training_dataset(
    csv_path: str | Path,
    feature_func_path: str,
    horizon: int = 16,
) -> TrainingDataset:
    """Load the CSV, build features and construct the training dataset."""

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    feature_builder = load_feature_builder(feature_func_path)
    features = feature_builder(raw)

    if "timestamp" in features.columns:
        timestamps = pd.to_datetime(features["timestamp"], utc=True, errors="coerce")
    else:
        timestamps = pd.Series(pd.RangeIndex(start=0, stop=len(features)), name="timestamp")

    returns = compute_future_returns(features, horizon)
    labels = make_labels(features, horizon)

    valid_mask = (~returns.isna()) & (~labels.isna())
    filtered = features.loc[valid_mask].reset_index(drop=True)
    returns = returns.loc[valid_mask].reset_index(drop=True)
    labels = labels.loc[valid_mask].reset_index(drop=True)
    timestamps = timestamps.loc[valid_mask].reset_index(drop=True)

    numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if not c.endswith("label")]

    dataset = TrainingDataset(
        features=filtered[feature_cols].copy(),
        labels=labels.astype(int),
        future_returns=returns,
        feature_columns=feature_cols,
        timestamps=timestamps,
        metadata={
            "csv_path": str(csv_path),
            "feature_func": feature_func_path,
            "horizon": horizon,
            "raw_rows": len(raw),
            "rows_after_filter": len(filtered),
        },
    )
    return dataset


def _build_estimator(model_params: Optional[Dict[str, Any]], random_state: int) -> Pipeline:
    params = dict(DEFAULT_MODEL_PARAMS)
    if model_params:
        params.update(model_params)
    params.setdefault("random_state", random_state)

    estimator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    **params,
                ),
            ),
        ]
    )
    return estimator


def _determine_cv_splits(n_samples: int) -> int:
    if n_samples < 50:
        return max(2, n_samples - 1)
    if n_samples < 500:
        return 3
    if n_samples < 1000:
        return 4
    return 5


def scan_thresholds(
    probabilities: Sequence[float],
    labels: Sequence[int],
    future_returns: Optional[Sequence[float]],
    thresholds: Iterable[float],
    maximize: str = "f1",
) -> Dict[str, Any]:
    """Evaluate a set of decision thresholds and pick the best one."""

    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probabilities)
    returns_arr = (
        np.asarray(future_returns)
        if future_returns is not None
        else np.full_like(probs_arr, np.nan, dtype=float)
    )

    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        mask = probs_arr >= thr
        selected = mask.sum()
        coverage = float(selected) / float(len(probs_arr)) if len(probs_arr) else 0.0
        if selected == 0:
            precision = recall = f1 = 0.0
            avg_return = float("nan")
            median_return = float("nan")
            win_rate = 0.0
            total_return = 0.0
        else:
            preds = mask.astype(int)
            precision = precision_score(labels_arr, preds, zero_division=0)
            recall = recall_score(labels_arr, preds, zero_division=0)
            f1 = f1_score(labels_arr, preds, zero_division=0)
            sub_returns = returns_arr[mask]
            valid_returns = sub_returns[np.isfinite(sub_returns)]
            if valid_returns.size:
                avg_return = float(np.nanmean(valid_returns))
                median_return = float(np.nanmedian(valid_returns))
                win_rate = float(np.mean(valid_returns > 0))
                total_return = float(np.prod(1.0 + valid_returns) - 1.0)
            else:
                avg_return = float("nan")
                median_return = float("nan")
                win_rate = 0.0
                total_return = 0.0
        rows.append(
            {
                "threshold": float(thr),
                "coverage": coverage,
                "signals": int(selected),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "avg_return": avg_return,
                "median_return": median_return,
                "win_rate": float(win_rate),
                "total_return": float(total_return),
            }
        )

    maximize_key = maximize if maximize in rows[0] else "f1"
    best_row = max(rows, key=lambda r: (r.get(maximize_key) or 0.0, r["threshold"]))
    return {"thresholds": rows, "best": best_row, "maximize": maximize_key}


def train_model(
    dataset: TrainingDataset,
    models_dir: str | Path,
    calibration: str = "isotonic",
    model_params: Optional[Dict[str, Any]] = None,
    threshold_grid: Optional[Iterable[float]] = None,
    random_state: int = 42,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Train the model, calibrate probabilities and evaluate thresholds."""

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X = dataset.features
    y = dataset.labels
    returns = dataset.future_returns

    if len(X) <= 5:
        raise ValueError("Dataset too small for training")

    cv_splits = _determine_cv_splits(len(X))
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    cv_scores: List[float] = []
    for train_idx, test_idx in tscv.split(X):
        est = _build_estimator(model_params, random_state)
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = est.predict_proba(X.iloc[test_idx])[:, 1]
        score = roc_auc_score(y.iloc[test_idx], proba)
        cv_scores.append(float(score))

    calibration_size = max(int(len(X) * 0.2), 200)
    if calibration_size >= len(X):
        calibration_size = max(1, len(X) // 5)
    train_end = len(X) - calibration_size
    if train_end < 1:
        train_end = max(1, int(len(X) * 0.7))
    if train_end >= len(X):
        train_end = len(X) - 1
    if train_end < 1:
        raise ValueError("Insufficient samples for calibration split")

    base_estimator = _build_estimator(model_params, random_state)
    base_estimator.fit(X.iloc[:train_end], y.iloc[:train_end])

    calibrator = CalibratedClassifierCV(
        base_estimator=base_estimator,
        method=calibration,
        cv="prefit",
    )
    calibrator.fit(X.iloc[train_end:], y.iloc[train_end:])

    proba_all = calibrator.predict_proba(X)[:, 1]
    preds_default = (proba_all >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y, proba_all)),
        "log_loss": float(log_loss(y, proba_all, eps=1e-9)),
        "accuracy@0.5": float(accuracy_score(y, preds_default)),
        "precision@0.5": float(precision_score(y, preds_default, zero_division=0)),
        "recall@0.5": float(recall_score(y, preds_default, zero_division=0)),
        "f1@0.5": float(f1_score(y, preds_default, zero_division=0)),
        "coverage@0.5": float(float(preds_default.mean())),
    }

    grid = (
        list(threshold_grid)
        if threshold_grid is not None
        else [round(x, 2) for x in np.linspace(0.4, 0.7, 16)]
    )
    threshold_summary = scan_thresholds(proba_all, y, returns, grid, maximize="f1")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_name or timestamp
    run_dir = models_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    joblib.dump(calibrator, model_path)

    metadata = {
        "timestamp": timestamp,
        "run_id": run_id,
        "calibration_method": calibration,
        "model_params": {**DEFAULT_MODEL_PARAMS, **(model_params or {})},
        "metrics": metrics,
        "cv_scores": cv_scores,
        "thresholds": threshold_summary,
        "train_size": int(train_end),
        "calibration_size": int(len(X) - train_end),
        "feature_columns": dataset.feature_columns,
        "dataset": dataset.metadata,
    }

    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    thresholds_path = run_dir / "thresholds.json"
    with thresholds_path.open("w", encoding="utf-8") as fh:
        json.dump(threshold_summary, fh, indent=2)

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "thresholds_path": str(thresholds_path),
        "run_dir": str(run_dir),
        "timestamp": timestamp,
        "run_id": run_id,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "threshold_summary": threshold_summary,
    }


def _parse_model_params(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse --model-params JSON: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the h16 workflow model")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument(
        "--feature-func",
        default="csp.features.h16.build_features_15m_4h",
        help="Dotted path to the feature engineering function",
    )
    parser.add_argument("--models-dir", default="models/ci", help="Output directory for models")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument(
        "--calibration",
        default="isotonic",
        choices=["isotonic", "sigmoid"],
        help="Probability calibration method",
    )
    parser.add_argument("--model-params", help="JSON string with model overrides", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-name", default=None, help="Optional run name")

    args = parser.parse_args()
    dataset = prepare_training_dataset(args.csv, args.feature_func, horizon=args.horizon)
    result = train_model(
        dataset,
        models_dir=args.models_dir,
        calibration=args.calibration,
        model_params=_parse_model_params(args.model_params),
        random_state=args.random_state,
        run_name=args.run_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
