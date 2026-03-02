from __future__ import annotations
from typing import Dict, Any
import pandas as pd

from csp.features.h16 import build_features_15m_4h
from csp.core.feature import add_features as _add_extra_features


def default_params() -> Dict[str, Any]:
    return {
        "ema_windows": (9, 21, 50),
        "rsi_window": 14,
        "bb_window": 20,
        "bb_std": 2.0,
        "atr_window": 14,
        "prev_high_period": 20,
        "prev_low_period": 20,
        "atr_percentile_window": 100,
        "h4_resample": "4H",
    }


def add_features(df: pd.DataFrame, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Add technical features to ``df`` using the given ``params``.

    Parameters are aligned with ``strategy.yaml`` and the existing
    training/realtime feature set.
    """
    p = default_params()
    if params:
        p.update(params)
    feats = build_features_15m_4h(
        df,
        ema_windows=tuple(p["ema_windows"]),
        rsi_window=int(p["rsi_window"]),
        bb_window=int(p["bb_window"]),
        bb_std=float(p["bb_std"]),
        atr_window=int(p["atr_window"]),
        h4_resample=p.get("h4_resample", "4H"),
    )
    feats = _add_extra_features(
        feats,
        prev_high_period=int(p["prev_high_period"]),
        prev_low_period=int(p["prev_low_period"]),
        bb_window=int(p["bb_window"]),
        atr_window=int(p["atr_window"]),
        atr_percentile_window=int(p["atr_percentile_window"]),
    )
    return feats
