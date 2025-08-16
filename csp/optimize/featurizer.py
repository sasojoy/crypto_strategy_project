from __future__ import annotations
from typing import Dict, Any
import pandas as pd

from csp.features.h16 import build_features_15m_4h


def default_params() -> Dict[str, Any]:
    return {
        "ema_windows": (9, 21, 50),
        "rsi_window": 14,
        "bb_window": 20,
        "bb_std": 2.0,
        "atr_window": 14,
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
    return feats
