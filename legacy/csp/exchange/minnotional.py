import os, json, time
from typing import Dict, Any
import requests

_CACHE_TTL_SEC = 3600
_CACHE_PATH = os.environ.get("MIN_NOTIONAL_CACHE", "/opt/crypto_strategy_project/resources/exchange_info_cache.json")

def _load_local_min_notional_from_cfg(cfg: Dict[str, Any], symbol: str) -> float | None:
    try:
        return float(cfg.get("trade", {}).get("min_notional", {}).get(symbol))
    except Exception:
        return None

def _load_cache() -> Dict[str, Any]:
    if not _CACHE_PATH or not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass

def _fetch_exchange_info() -> Dict[str, Any]:
    url = "https://api.binance.com/api/v3/exchangeInfo"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def _parse_min_notional(exchange_info: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s in exchange_info.get("symbols", []):
        sym = s.get("symbol")
        mins = []
        for f in s.get("filters", []):
            if f.get("filterType") == "MIN_NOTIONAL":
                try:
                    mins.append(float(f.get("minNotional")))
                except Exception:
                    pass
        if sym and mins:
            out[sym] = max(mins)
    return out

def get_min_notional(symbol: str, cfg: Dict[str, Any]) -> float:
    """
    取得某 symbol 的最小下單名目（USDT）。
    優先順序：config.trade.min_notional[symbol] → 本地快取 → 線上拉取後寫入快取。
    若任何步驟失敗，回傳 0.0（不阻擋下單；在上層 guard 會印出 warning）。
    """
    v = _load_local_min_notional_from_cfg(cfg, symbol)
    if v is not None:
        return float(v)
    cache = _load_cache()
    now = time.time()
    if cache and (now - float(cache.get("_ts", 0))) <= _CACHE_TTL_SEC:
        v = cache.get("min_notional", {}).get(symbol)
        if v is not None:
            return float(v)
    try:
        ei = _fetch_exchange_info()
        table = _parse_min_notional(ei)
        new_cache = {"_ts": now, "min_notional": table}
        _save_cache(new_cache)
        return float(table.get(symbol, 0.0))
    except Exception:
        return 0.0
