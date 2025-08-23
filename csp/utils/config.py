from __future__ import annotations
from typing import Dict, Any


def get_symbol_features(cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """Return merged feature parameters for ``symbol``.

    Configuration format::

        features:
          default:
            rsi: {enabled: true, window: 14}
            bollinger: {enabled: true, window: 20, std: 2.0}
            atr: {enabled: true, window: 14}
          per_symbol:
            BTCUSDT:
              rsi: {enabled: true, window: 20}
              ...

    The function first reads ``features.default`` then overrides with
    ``features.per_symbol.<SYMBOL>`` if present.  The returned dictionary
    is flattened to match the historical ``feature`` parameters used by the
    feature builders.
    """
    assert isinstance(cfg, dict), f"cfg must be dict, got {type(cfg)}"
    feats = cfg.get("features", {})
    default = feats.get("default", {})
    per_sym = feats.get("per_symbol", {}).get(symbol, {})

    def merge(name: str) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        if isinstance(default.get(name), dict):
            res.update(default[name])
        if isinstance(per_sym.get(name), dict):
            res.update(per_sym[name])
        return res

    out: Dict[str, Any] = {}
    # parameters used by build_features_15m_4h
    out["ema_windows"] = default.get("ema_windows", (9, 21, 50))
    h4_rule = default.get("h4_rule", {"resample": "4H"})
    out["h4_rule"] = h4_rule
    out["h4_resample"] = h4_rule.get("resample", "4H")

    rsi = merge("rsi")
    boll = merge("bollinger")
    atr = merge("atr")

    bb_window_base = int(boll.get("window", 20))
    atr_window_base = int(atr.get("window", 14))
    bb_window = int(feats.get("bb_window", bb_window_base))
    atr_window = int(feats.get("atr_window", atr_window_base))
    bb_window = int(per_sym.get("bb_window", bb_window))
    atr_window = int(per_sym.get("atr_window", atr_window))

    prev_high_period = int(feats.get("prev_high_period", 20))
    prev_low_period = int(feats.get("prev_low_period", 20))
    atr_pct_window = int(feats.get("atr_percentile_window", 100))
    prev_high_period = int(per_sym.get("prev_high_period", prev_high_period))
    prev_low_period = int(per_sym.get("prev_low_period", prev_low_period))
    atr_pct_window = int(per_sym.get("atr_percentile_window", atr_pct_window))

    out.update({
        "rsi_window": int(rsi.get("window", 14)),
        "bb_window": bb_window,
        "bb_std": float(boll.get("std", 2.0)),
        "atr_window": atr_window,
        "prev_high_period": prev_high_period,
        "prev_low_period": prev_low_period,
        "atr_percentile_window": atr_pct_window,
    })
    return out
