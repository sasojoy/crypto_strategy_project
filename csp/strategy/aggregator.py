from __future__ import annotations

"""Utilities for aggregating multi-horizon model probabilities and fetching
the latest trading signal.

This module adds several safety guards to ensure the produced signal is
well-formed:

* ``aggregate_signal`` sanitises the probability map to remove ``NaN`` or
  out-of-range values and always returns a non-``NaN`` score.
* ``get_latest_signal`` loads the latest CSV data, checks data freshness and
  verifies that model ``feature_columns`` align with the real-time features.
  Problems yield ``None`` so the caller can guard against stale or malformed
  inputs.
"""

import json
import math
import os
from typing import Dict, Optional

import pandas as pd


def _weight(h: int) -> float:
    """Return weight for horizon ``h``.

    The default implementation uses ``sqrt`` so that longer horizons are given
    slightly more importance without overwhelming shorter horizons.
    """

    return math.sqrt(max(1, int(h)))


def _clean_prob_map(prob_map: dict) -> dict:
    """Remove ``NaN`` or out-of-range probabilities from ``prob_map``."""

    clean: Dict = {}
    for k, v in (prob_map or {}).items():
        try:
            f = float(v)
        except Exception:
            continue
        if math.isnan(f) or f < 0.0 or f > 1.0:
            continue
        clean[k] = f
    return clean


def aggregate_signal(prob_map: dict, enter_threshold: float = 0.75, method: str = "max_weighted") -> dict:
    """Aggregate ``prob_map`` into a single trading decision.

    Parameters
    ----------
    prob_map : dict
        Mapping ``(horizon, threshold) -> probability_of_up``.
    enter_threshold : float, optional
        Minimum probability required to enter a trade.
    method : str, optional
        Aggregation method. ``"majority"`` or ``"max_weighted"`` (default).

    Returns
    -------
    dict
        Contains at least ``side`` (``LONG``/``SHORT``/``NONE``), ``score``
        (never ``NaN``), ``prob_up_max``, ``prob_down_max``, ``chosen_h``,
        ``chosen_t`` and ``reason``.
    """

    clean = _clean_prob_map(prob_map)
    if not clean:
        return {
            "side": "NONE",
            "score": 0.0,
            "prob_up_max": 0.0,
            "prob_down_max": 1.0,
            "chosen_h": None,
            "chosen_t": None,
            "reason": "empty_or_nan_prob_map",
        }

    if method == "majority":
        ups = sum(1 for _, p in clean.items() if p >= enter_threshold)
        downs = sum(1 for _, p in clean.items() if (1.0 - p) >= enter_threshold)
        if ups > downs and ups > 0:
            side = "LONG"
            score = 1.0
        elif downs > ups and downs > 0:
            side = "SHORT"
            score = 1.0
        else:
            side = "NONE"
            score = 0.0
        prob_up_max = max(clean.values())
        return {
            "side": side,
            "score": float(score),
            "prob_up_max": float(prob_up_max),
            "prob_down_max": float(1.0 - prob_up_max),
            "chosen_h": None,
            "chosen_t": None,
            "reason": "majority",
        }

    # default: max_weighted
    scored = [((h, t), p * _weight(h)) for (h, t), p in clean.items()]
    if not scored:
        return {
            "side": "NONE",
            "score": 0.0,
            "prob_up_max": 0.0,
            "prob_down_max": 1.0,
            "chosen_h": None,
            "chosen_t": None,
            "reason": "scored_empty",
        }
    (chosen_ht, score) = max(scored, key=lambda x: x[1])
    (ch, ct) = chosen_ht
    prob_up_max = max(clean.values())
    side = "LONG" if prob_up_max >= enter_threshold else "NONE"
    safe_score = 0.0 if (score is None or math.isnan(score)) else float(score)
    return {
        "side": side,
        "score": safe_score,
        "prob_up_max": float(prob_up_max),
        "prob_down_max": float(1.0 - prob_up_max),
        "chosen_h": int(ch),
        "chosen_t": float(ct),
        "reason": "ok" if side != "NONE" else "below_threshold",
    }


def get_latest_signal(symbol: str, cfg: dict, fresh_min: float = 5.0) -> Optional[dict]:
    """Load the latest data and model for ``symbol`` and return aggregated signal.

    The function performs several checks:

    * CSV path existence and required timestamp column.
    * Data freshness not older than ``fresh_min`` minutes.
    * Model ``feature_columns`` must all exist in the engineered features.
    * No ``NaN`` values in the input features for prediction.

    Returns ``None`` if any check fails.
    """

    from csp.core.feature import add_features  # local import to keep dependency light

    try:
        from csp.models.classifier_multi import MultiThresholdClassifier
    except Exception:
        return None

    csv_path = cfg["io"]["csv_paths"].get(symbol)
    if not csv_path or not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        return None
    ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
    age_min = (pd.Timestamp.utcnow() - ts).total_seconds() / 60.0
    if age_min > fresh_min:
        # data too old
        return None

    dff = add_features(df.copy())
    model_dir = os.path.join(cfg["io"].get("models_dir", "models"), symbol, "cls_multi")
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    feat_cols = meta.get("feature_columns") or []
    if not feat_cols:
        return None
    for c in feat_cols:
        if c not in dff.columns:
            return None

    X = dff[feat_cols].tail(1)
    if X.isna().any().any():
        return None

    m = MultiThresholdClassifier.load(model_dir)
    prob_map = m.predict_proba(X)
    th = cfg.get("strategy", {}).get("enter_threshold", 0.75)
    method = cfg.get("strategy", {}).get("aggregator_method", "max_weighted")
    sig = aggregate_signal(prob_map, enter_threshold=th, method=method)

    sig["symbol"] = symbol
    sig["ts"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    # expose latest ATR for sizing if available
    try:
        if "atr" in dff.columns:
            sig["atr_abs"] = float(dff["atr"].iloc[-1])
        elif "atr_h4" in dff.columns:
            sig["atr_abs"] = float(dff["atr_h4"].iloc[-1])
    except Exception:
        pass
    return sig


__all__ = ["aggregate_signal", "get_latest_signal"]

