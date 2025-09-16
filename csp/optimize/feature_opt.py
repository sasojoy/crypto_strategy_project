from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List, Any

import optuna
import pandas as pd
import yaml
import shutil
import datetime
import difflib

from .evaluator import walk_forward_evaluate
from csp.utils.config import get_symbol_features


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
        "prev_high_period": trial.suggest_int("prev_high_period", 5, 60),
        "prev_low_period": trial.suggest_int("prev_low_period", 5, 60),
        "atr_percentile_window": trial.suggest_int("atr_percentile_window", 50, 200),
    }


def optimize_symbol(cfg_path: str, symbol: str,
                    start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                    n_trials: int = 20, out_dir: Path | None = None,
                    timeout: int | None = None) -> optuna.Study:
    """Run Optuna optimization for a single symbol."""
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    base_feature = get_symbol_features(cfg, symbol)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        # merge with base feature params (e.g., ema_windows)
        merged = {**base_feature, **params}
        res = walk_forward_evaluate(cfg_path, symbol, start_ts, end_ts, merged)
        metric = res.get("metrics", {}).get("總報酬率%", 0.0)
        trial.set_user_attr("metrics", res.get("metrics", {}))
        return float(metric)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

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
                     n_trials: int = 20, out_dir: Path | None = None,
                     timeout: int | None = None) -> Dict[str, optuna.Study]:
    results: Dict[str, optuna.Study] = {}
    for sym in symbols:
        study = optimize_symbol(cfg_path, sym, start_ts, end_ts, n_trials, out_dir, timeout)
        results[sym] = study
    return results


def apply_best_params_to_cfg(cfg_path: str, symbol: str, best_params: Dict[str, Any],
                             *, apply: bool = False, log_file: Path | None = None) -> None:
    """Update ``cfg_path`` with ``best_params`` for ``symbol``.

    A backup ``strategy.yaml.bak-YYYYmmdd-HHMMSS`` is created before overwriting.
    The diff of the change is printed and optionally appended to ``log_file``.
    """
    if not apply:
        return
    if not best_params:
        print(f"[SKIP] {symbol}: no best params to apply")
        return

    cfg_path = Path(cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    feats = cfg.setdefault("features", {})
    feats.setdefault("default", {})
    per = feats.setdefault("per_symbol", {})
    sym_cfg = per.setdefault(symbol, {})

    # ensure sub-structures exist
    sym_cfg.setdefault("rsi", {})
    sym_cfg.setdefault("bollinger", {})
    sym_cfg.setdefault("atr", {})
    sym_cfg["rsi"].update({"enabled": True, "window": int(best_params["rsi_window"])})
    sym_cfg["bollinger"].update({
        "enabled": True,
        "window": int(best_params["bb_window"]),
        "std": float(best_params["bb_std"]),
    })
    sym_cfg["atr"].update({"enabled": True, "window": int(best_params["atr_window"])})
    sym_cfg["prev_high_period"] = int(best_params.get("prev_high_period", 20))
    sym_cfg["prev_low_period"] = int(best_params.get("prev_low_period", 20))
    sym_cfg["atr_percentile_window"] = int(best_params.get("atr_percentile_window", 100))

    old_text = cfg_path.read_text(encoding="utf-8")
    new_text = yaml.dump(cfg, allow_unicode=True, sort_keys=False)
    if old_text == new_text:
        print(f"[INFO] {symbol}: config unchanged")
        return

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = cfg_path.with_suffix(cfg_path.suffix + f".bak-{ts}")
    shutil.copy(cfg_path, backup_path)
    cfg_path.write_text(new_text, encoding="utf-8")

    diff = "".join(difflib.unified_diff(
        old_text.splitlines(True), new_text.splitlines(True),
        fromfile="before", tofile="after"
    ))
    print(diff)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"# {symbol}\n{diff}\n")
