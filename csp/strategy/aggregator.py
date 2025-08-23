from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import pandas as pd

from csp.models.classifier_multi import MultiThresholdClassifier
from csp.features.h16 import build_features_15m_4h
from csp.core.feature import add_features
from csp.utils.config import get_symbol_features


def _weight(h: int, fn: str = "sqrt") -> float:
    """Return weight for horizon ``h`` according to ``fn``."""
    h = max(h, 1)
    if fn == "log":
        return math.log(h)
    if fn == "linear":
        return float(h)
    return math.sqrt(h)


def aggregate_signal(
    prob_map: Dict[Tuple[int, float], float],
    enter_threshold: float = 0.75,
    method: str = "max_weighted",
    weight_fn: str = "sqrt",
) -> Dict[str, Any]:
    """Aggregate (h, t)->probability map into a single trading signal.

    Parameters
    ----------
    prob_map : dict
        Mapping from ``(horizon, threshold)`` to ``p_up``.
    enter_threshold : float, optional
        Minimum score/probability required to enter a trade.
    method : str, optional
        ``"max_weighted"`` or ``"majority"``.
    weight_fn : str, optional
        Weighting function for ``max_weighted`` method (``sqrt``/``log``/``linear``).

    Returns
    -------
    dict
        Aggregated signal information.
    """
    if not prob_map:
        return {
            "side": "NONE",
            "score": 0.0,
            "prob_up_max": 0.0,
            "prob_down_max": 0.0,
            "chosen_h": None,
            "chosen_t": None,
        }

    prob_up_max = max(prob_map.values())
    prob_down_max = max(1.0 - p for p in prob_map.values())

    if method == "majority":
        long_cnt = sum(p >= 0.5 for p in prob_map.values())
        short_cnt = len(prob_map) - long_cnt
        if long_cnt > short_cnt and prob_up_max >= enter_threshold:
            chosen_h, chosen_t = max(prob_map.items(), key=lambda kv: kv[1])[0]
            return {
                "side": "LONG",
                "score": prob_up_max,
                "prob_up_max": prob_up_max,
                "prob_down_max": prob_down_max,
                "chosen_h": chosen_h,
                "chosen_t": chosen_t,
            }
        if short_cnt > long_cnt and prob_down_max >= enter_threshold:
            chosen_h, chosen_t = max(prob_map.items(), key=lambda kv: 1.0 - kv[1])[0]
            return {
                "side": "SHORT",
                "score": prob_down_max,
                "prob_up_max": prob_up_max,
                "prob_down_max": prob_down_max,
                "chosen_h": chosen_h,
                "chosen_t": chosen_t,
            }
        return {
            "side": "NONE",
            "score": 0.0,
            "prob_up_max": prob_up_max,
            "prob_down_max": prob_down_max,
            "chosen_h": None,
            "chosen_t": None,
        }

    # default: max_weighted
    best_long: Tuple[float, float, int, float] | None = None  # score, p_up, h, t
    best_short: Tuple[float, float, int, float] | None = None
    for (h, t), p in prob_map.items():
        w = _weight(h, weight_fn)
        l_score = p * w
        s_score = (1.0 - p) * w
        if best_long is None or l_score > best_long[0]:
            best_long = (l_score, p, h, t)
        if best_short is None or s_score > best_short[0]:
            best_short = (s_score, 1.0 - p, h, t)

    side = "NONE"
    score = 0.0
    chosen_h = chosen_t = None
    if best_long and best_long[0] >= enter_threshold and (
        not best_short or best_long[0] >= best_short[0]
    ):
        side = "LONG"
        score = best_long[0]
        chosen_h, chosen_t = best_long[2], best_long[3]
    elif best_short and best_short[0] >= enter_threshold and (
        not best_long or best_short[0] > best_long[0]
    ):
        side = "SHORT"
        score = best_short[0]
        chosen_h, chosen_t = best_short[2], best_short[3]

    return {
        "side": side,
        "score": float(score),
        "prob_up_max": float(prob_up_max),
        "prob_down_max": float(prob_down_max),
        "chosen_h": chosen_h,
        "chosen_t": chosen_t,
    }


def get_latest_signal(symbol: str, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load latest features and model for ``symbol`` and return aggregated signal."""
    try:
        csv_path = cfg.get("io", {}).get("csv_paths", {}).get(symbol)
        if not csv_path:
            return None
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        feat_params = get_symbol_features(cfg, symbol)
        feats = build_features_15m_4h(
            df,
            ema_windows=tuple(feat_params["ema_windows"]),
            rsi_window=feat_params["rsi_window"],
            bb_window=feat_params["bb_window"],
            bb_std=feat_params["bb_std"],
            atr_window=feat_params["atr_window"],
            h4_resample=feat_params.get("h4_resample", "4H"),
        )
        feats = add_features(
            feats,
            prev_high_period=feat_params["prev_high_period"],
            prev_low_period=feat_params["prev_low_period"],
            bb_window=feat_params["bb_window"],
            atr_window=feat_params["atr_window"],
            atr_percentile_window=feat_params["atr_percentile_window"],
        )
        latest = feats.tail(1)
        if latest.empty:
            return None
        models_dir = cfg.get("io", {}).get("models_dir", "models")
        mdir = Path(models_dir) / symbol / "cls_multi"
        if not mdir.exists():
            return None
        clf = MultiThresholdClassifier.load(str(mdir))
        prob_map = clf.predict_proba(latest)
        strat = cfg.get("strategy", {})
        enter_thr = float(strat.get("enter_threshold", 0.75))
        method = strat.get("aggregator_method", "max_weighted")
        weight_fn = strat.get("weight_fn", "sqrt")
        agg = aggregate_signal(prob_map, enter_thr, method, weight_fn)
        ts = latest["timestamp"].iloc[-1]
        ts = ts.tz_convert(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        agg.update({"symbol": symbol, "ts": ts.isoformat().replace("+00:00", "Z")})
        try:
            # expose latest H4 ATR for sizing
            agg["atr_abs"] = float(latest.get("atr_h4", latest.get("atr", 0.0)))
        except Exception:
            pass
        return agg
    except Exception:
        return None
