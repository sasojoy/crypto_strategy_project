from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List

import optuna
import pandas as pd
import yaml

from .evaluator import walk_forward_evaluate


def _suggest_params(trial: optuna.Trial) -> Dict:
    """Search space for feature parameters.

    EMA windows are kept fixed to avoid mismatch with trained models.
    """
    return {
        # keep ema_windows from config
        "rsi_window": trial.suggest_int("rsi_window", 5, 30),
        "bb_window": trial.suggest_int("bb_window", 10, 40),
        "bb_std": trial.suggest_float("bb_std", 1.0, 3.5),
        "atr_window": trial.suggest_int("atr_window", 7, 40),
    }


def optimize_symbol(cfg_path: str, symbol: str,
                    start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                    n_trials: int = 20, out_dir: Path | None = None) -> optuna.Study:
    """Run Optuna optimization for a single symbol."""
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    base_feature = cfg.get("feature", {})

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        # merge with base feature params (e.g., ema_windows)
        merged = {**base_feature, **params}
        res = walk_forward_evaluate(cfg_path, symbol, start_ts, end_ts, merged)
        metric = res.get("metrics", {}).get("總報酬率%", 0.0)
        trial.set_user_attr("metrics", res.get("metrics", {}))
        return float(metric)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # trials dataframe
        df = study.trials_dataframe()
        df.to_csv(out_dir / f"{symbol}_trials.csv", index=False, encoding="utf-8-sig")
        # best summary
        summary = {
            "best_value": study.best_value,
            "best_params": study.best_params,
        }
        (out_dir / f"{symbol}_best.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return study


def optimize_symbols(cfg_path: str, symbols: List[str],
                     start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                     n_trials: int = 20, out_dir: Path | None = None) -> Dict[str, optuna.Study]:
    results: Dict[str, optuna.Study] = {}
    for sym in symbols:
        study = optimize_symbol(cfg_path, sym, start_ts, end_ts, n_trials, out_dir)
        results[sym] = study
    return results
