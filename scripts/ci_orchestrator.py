"""CI orchestrator for the model workflow.

This script ties together training, backtesting style threshold evaluation and
notification logic.  It is designed to be invoked from GitHub Actions but can
also be run locally for debugging.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from scripts.notify_telegram import send_telegram_message
except Exception:  # noqa: BLE001
    import sys

    sys.path.append(os.path.dirname(__file__))
    from notify_telegram import send_telegram_message  # type: ignore[import-not-found]
from scripts.train_h16_wf import (
    TrainingDataset,
    prepare_training_dataset,
    train_model,
)


def _parse_thresholds(raw: Optional[str]) -> List[float]:
    if not raw:
        return [round(x, 2) for x in [0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]]
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


def _default_search_space() -> List[Dict[str, Any]]:
    return [
        {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 600},
        {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.9},
        {"max_depth": 3, "learning_rate": 0.07, "colsample_bytree": 0.9},
    ]


def _should_notify(mode: str, success: bool) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return not success


def _compose_message(
    success: bool,
    target_metric: str,
    target_value: float,
    best_run: Optional[Dict[str, Any]],
    ci_log_path: Path,
) -> str:
    status_icon = "✅" if success else "⚠️"
    lines = [f"{status_icon} Model CI"]
    lines.append(f"Target: {target_metric} ≥ {target_value:.3f}")
    if best_run:
        metrics = best_run.get("metrics", {})
        value = metrics.get(target_metric)
        lines.append(
            f"Best run {best_run.get('run_id')} → {target_metric}={value:.3f}" if value is not None else "Best run available"
        )
        best_threshold = (best_run.get("threshold_summary") or {}).get("best", {})
        if best_threshold:
            lines.append(
                "Threshold {threshold:.2f} | coverage {coverage:.1%} | precision {precision:.2f}".format(
                    threshold=best_threshold.get("threshold", 0.5),
                    coverage=best_threshold.get("coverage", 0.0),
                    precision=best_threshold.get("precision", 0.0),
                )
            )
        run_dir = best_run.get("run_dir")
        if run_dir:
            lines.append(f"Model dir: {run_dir}")
    lines.append(f"Log: {ci_log_path}")
    if not success:
        lines.append("Please attach logs/ci_run.json to ChatGPT for follow-up analysis.")
    return "\n".join(lines)


def run_attempt(
    attempt_id: int,
    dataset: TrainingDataset,
    models_dir: Path,
    calibration: str,
    model_params: Optional[Dict[str, Any]],
    thresholds: Iterable[float],
    random_state: int,
) -> Dict[str, Any]:
    run_name = f"ci_run_{attempt_id:02d}"
    result = train_model(
        dataset,
        models_dir=models_dir,
        calibration=calibration,
        model_params=model_params,
        threshold_grid=thresholds,
        random_state=random_state + attempt_id,
        run_name=run_name,
    )
    result["attempt"] = attempt_id
    result["model_params"] = model_params or {}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CI orchestrator for model workflow")
    parser.add_argument("--csv", default="resources/btc_15m.csv")
    parser.add_argument(
        "--feature-func",
        default="csp.features.h16.build_features_15m_4h",
        help="Dotted path to feature function",
    )
    parser.add_argument("--models-dir", default="models/ci")
    parser.add_argument("--artifacts-dir", default="artifacts/ci")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--calibration", default="isotonic", choices=["isotonic", "sigmoid"])
    parser.add_argument("--target-metric", default="roc_auc")
    parser.add_argument("--target-value", type=float, default=0.55)
    parser.add_argument("--max-search", type=int, default=3)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--notify", default="on_fail", choices=["always", "on_fail", "never"])
    parser.add_argument("--telegram-token", default=None)
    parser.add_argument("--telegram-chat-id", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default="logs/ci_run.json")

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    artifacts_dir = Path(args.artifacts_dir)
    logs_path = Path(args.output)
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_thresholds(args.thresholds)

    ci_log: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "csv": args.csv,
        "feature_func": args.feature_func,
        "target_metric": args.target_metric,
        "target_value": args.target_value,
        "attempts": [],
        "success": False,
        "error": None,
    }

    dataset: Optional[TrainingDataset] = None
    best_run: Optional[Dict[str, Any]] = None

    try:
        dataset = prepare_training_dataset(args.csv, args.feature_func, horizon=args.horizon)
        search_space = [None] + _default_search_space()[: max(args.max_search - 1, 0)]
        for attempt_id, params in enumerate(search_space, start=1):
            result = run_attempt(
                attempt_id,
                dataset,
                models_dir,
                args.calibration,
                params,
                thresholds,
                args.random_state,
            )
            metric_value = result["metrics"].get(args.target_metric)
            result["metric_value"] = metric_value
            result["meets_target"] = bool(metric_value is not None and metric_value >= args.target_value)
            ci_log["attempts"].append(result)
            if best_run is None or (
                metric_value is not None and metric_value > best_run.get("metric_value", float("-inf"))
            ):
                best_run = result
            if result["meets_target"]:
                break
    except Exception as exc:  # noqa: BLE001
        ci_log["error"] = {
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    if best_run:
        ci_log["best"] = best_run
        ci_log["success"] = bool(best_run.get("metric_value") and best_run["metric_value"] >= args.target_value)
    else:
        ci_log["best"] = None
        ci_log["success"] = False

    with logs_path.open("w", encoding="utf-8") as fh:
        json.dump(ci_log, fh, indent=2)

    if best_run:
        best_summary_path = logs_path.parent / "threshold_report.json"
        with best_summary_path.open("w", encoding="utf-8") as fh:
            json.dump(best_run.get("threshold_summary", {}), fh, indent=2)
        artifact_best_path = artifacts_dir / "best_run.json"
        with artifact_best_path.open("w", encoding="utf-8") as fh:
            json.dump(best_run, fh, indent=2)

    token = args.telegram_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = args.telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id and _should_notify(args.notify, ci_log["success"]):
        message = _compose_message(ci_log["success"], args.target_metric, args.target_value, best_run, logs_path)
        try:
            send_telegram_message(message, token=token, chat_id=chat_id)
        except Exception:  # noqa: BLE001
            # We do not want notification failures to fail the CI job.
            print("Failed to send Telegram notification:")
            traceback.print_exc()

    print(json.dumps(ci_log, indent=2))


if __name__ == "__main__":
    main()
