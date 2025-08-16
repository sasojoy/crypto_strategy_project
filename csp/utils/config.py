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

    out.update({
        "rsi_window": int(rsi.get("window", 14)),
        "bb_window": int(boll.get("window", 20)),
        "bb_std": float(boll.get("std", 2.0)),
        "atr_window": int(atr.get("window", 14)),
    })
    return out
