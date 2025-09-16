from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from csp.data.loader import load_15m_csv
from csp.optimize.feature_opt import optimize_symbol
from csp.utils.dates import resolve_time_range_like
from csp.utils.io import load_cfg
from csp.utils.tz_safe import normalize_df_to_utc

from .train import train_for_symbol


def _read_date_args_from_env() -> Dict[str, Any]:
    start = os.getenv("START_DATE")
    end = os.getenv("END_DATE")
    days = os.getenv("DAYS")
    if days is not None:
        try:
            days = int(days)
        except Exception:
            days = None
    return {"start": start, "end": end, "days": days}


def _apply_best_params(cfg: Dict[str, Any], symbol: str, best_params: Dict[str, Any]) -> None:
    if not best_params:
        return

    feats = cfg.setdefault("features", {})
    feats.setdefault("default", {})
    per_sym = feats.setdefault("per_symbol", {})
    sym_cfg: Dict[str, Any] = per_sym.setdefault(symbol, {})

    rsi_cfg = sym_cfg.setdefault("rsi", {})
    rsi_cfg.update({"enabled": True, "window": int(best_params.get("rsi_window", rsi_cfg.get("window", 14)))})

    boll_cfg = sym_cfg.setdefault("bollinger", {})
    boll_cfg.update({
        "enabled": True,
        "window": int(best_params.get("bb_window", boll_cfg.get("window", 20))),
        "std": float(best_params.get("bb_std", boll_cfg.get("std", 2.0))),
    })

    atr_cfg = sym_cfg.setdefault("atr", {})
    atr_cfg.update({
        "enabled": True,
        "window": int(best_params.get("atr_window", atr_cfg.get("window", 14))),
    })

    if "prev_high_period" in best_params:
        sym_cfg["prev_high_period"] = int(best_params["prev_high_period"])
    if "prev_low_period" in best_params:
        sym_cfg["prev_low_period"] = int(best_params["prev_low_period"])
    if "atr_percentile_window" in best_params:
        sym_cfg["atr_percentile_window"] = int(best_params["atr_percentile_window"])


def optimize_then_train_symbol(symbol: str, out_dir: str | Path,
                               cfg: Dict[str, Any] | str,
                               *,
                               n_trials: int = 50,
                               timeout_min: int | None = 20,
                               cfg_path: str | None = None) -> Dict[str, Any]:
    cfg_dict = load_cfg(cfg)
    cfg_path = cfg_path or (cfg if isinstance(cfg, str) else None)
    if not cfg_path:
        raise ValueError("cfg_path is required for optimization")

    csv_map = cfg_dict.get("io", {}).get("csv_paths", {})
    csv_path = csv_map.get(symbol)
    if not csv_path:
        raise ValueError(f"[OPT] {symbol}: csv path not configured")

    df = load_15m_csv(csv_path)
    df = normalize_df_to_utc(df)
    if df.empty:
        raise ValueError(f"[OPT] {symbol}: no data available for optimization")

    env_args = _read_date_args_from_env()
    start_ts, end_ts = resolve_time_range_like(env_args, df.index)

    opt_root = Path(out_dir) / "_optimize"
    opt_root.mkdir(parents=True, exist_ok=True)
    timeout_sec = None
    if timeout_min is not None and timeout_min > 0:
        timeout_sec = int(timeout_min * 60)

    print(
        f"[OPT] {symbol} window=({start_ts} ~ {end_ts}) trials={n_trials} timeout_sec={timeout_sec}"
    )
    study = optimize_symbol(
        cfg_path,
        symbol,
        start_ts=start_ts,
        end_ts=end_ts,
        n_trials=int(n_trials),
        timeout=timeout_sec,
        out_dir=opt_root,
    )
    best_params = dict(study.best_params)
    print(f"[OPT][BEST] {symbol} value={study.best_value:.6f} params={best_params}")

    _apply_best_params(cfg_dict, symbol, best_params)

    result = train_for_symbol(symbol, out_dir, cfg_dict, cfg_path=cfg_path)
    result["best_params"] = best_params
    result["optimize"] = {
        "best_value": float(study.best_value),
        "n_trials": int(len(study.trials)),
        "timeout_sec": timeout_sec,
        "start_ts": str(start_ts),
        "end_ts": str(end_ts),
        "artifact_dir": str(opt_root),
    }
    return result
