"""Threshold scanning and backtest style reporting for the CI workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np

from scripts.train_h16_wf import (
    TrainingDataset,
    prepare_training_dataset,
    scan_thresholds,
)


def _parse_thresholds(raw: Optional[str]) -> List[float]:
    if not raw:
        return [round(x, 2) for x in np.linspace(0.4, 0.7, 16)]
    values: List[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError as exc:
            raise SystemExit(f"Invalid threshold value: {chunk}") from exc
    return values


def run_threshold_report(
    model_path: str | Path,
    dataset: TrainingDataset,
    thresholds: Iterable[float],
    maximize: str = "f1",
) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not implement predict_proba")

    probabilities = model.predict_proba(dataset.features)[:, 1]
    summary = scan_thresholds(
        probabilities,
        dataset.labels,
        dataset.future_returns,
        thresholds,
        maximize=maximize,
    )
    summary["model_path"] = str(model_path)
    summary["dataset"] = dataset.metadata
    summary["rows"] = len(dataset.features)
    summary["threshold_grid"] = list(thresholds)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate threshold report")
    parser.add_argument("--model", required=True, help="Path to trained model.joblib")
    parser.add_argument("--csv", required=True, help="Dataset CSV path")
    parser.add_argument(
        "--feature-func",
        default="csp.features.h16.build_features_15m_4h",
        help="Dotted path to feature builder",
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--thresholds", default=None, help="Comma separated threshold list")
    parser.add_argument("--maximize", default="f1")
    parser.add_argument(
        "--output",
        default="logs/threshold_report.json",
        help="Where to write the JSON report",
    )

    args = parser.parse_args()
    dataset = prepare_training_dataset(args.csv, args.feature_func, horizon=args.horizon)
    thresholds = _parse_thresholds(args.thresholds)
    summary = run_threshold_report(args.model, dataset, thresholds, maximize=args.maximize)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
