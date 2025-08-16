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
    cfg2.setdefault("feature", {}).update(feature_params)

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
