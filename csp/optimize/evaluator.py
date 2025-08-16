from __future__ import annotations
import os
import tempfile
from copy import deepcopy
from typing import Dict, Any

import pandas as pd
import yaml

from csp.backtesting.backtest_v2 import run_backtest_for_symbol


def walk_forward_evaluate(cfg_path: str, symbol: str,
                          start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                          feature_params: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate ``feature_params`` by running the existing backtest logic.

    A temporary config file is created with the feature parameters injected
    so that ``run_backtest_for_symbol`` can remain unchanged.
    """
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    cfg2 = deepcopy(cfg)
    feats = cfg2.setdefault("features", {})
    default = feats.setdefault("default", {})
    per = feats.setdefault("per_symbol", {})
    sym_cfg = per.setdefault(symbol, {})
    sym_cfg.setdefault("rsi", {})
    sym_cfg.setdefault("bollinger", {})
    sym_cfg.setdefault("atr", {})
    sym_cfg["rsi"]["window"] = int(feature_params["rsi_window"])
    sym_cfg["bollinger"]["window"] = int(feature_params["bb_window"])
    sym_cfg["bollinger"]["std"] = float(feature_params["bb_std"])
    sym_cfg["atr"]["window"] = int(feature_params["atr_window"])
    # base parameters
    default["ema_windows"] = feature_params.get("ema_windows", default.get("ema_windows", (9, 21, 50)))
    h4_rule = default.setdefault("h4_rule", {})
    h4_rule["resample"] = feature_params.get("h4_resample", h4_rule.get("resample", "4H"))

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
        yaml.safe_dump(cfg2, tmp, allow_unicode=True)
        tmp_path = tmp.name
    try:
        csv_path = cfg2["io"]["csv_paths"][symbol]
        res = run_backtest_for_symbol(
            csv_path,
            tmp_path,
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    finally:
        os.unlink(tmp_path)
    return res
