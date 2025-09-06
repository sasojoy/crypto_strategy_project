from __future__ import annotations

import sys, math, traceback, os, json, logging, functools
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import tz

from csp.data.binance import fetch_klines_range
from csp.utils.timez import (
    ensure_utc_index,
    last_closed_15m,
    safe_ts_to_utc,
    ensure_aware_utc,
    now_utc,
)
from csp.utils.diag import log_diag, log_trace
from csp.utils.framefix import safe_reset_index
from csp.utils.paths import resolve_resources_dir


TZ_TW = tz.gettz("Asia/Taipei")
logger = logging.getLogger(__name__)


def _select_fetcher(cfg: Dict[str, Any]) -> Optional[callable]:
    rt = (cfg or {}).get("realtime", {})
    mode = str(rt.get("fetch", "")).lower()
    # TODO: wire actual live fetcher here if needed
    if mode in ("", "none", "csv_only"):
        return None
    return None


def _csv_path_for_symbol(symbol: str, cfg: Dict[str, Any]) -> Optional[str]:
    m = {
        "BTCUSDT": "btc_15m.csv",
        "ETHUSDT": "eth_15m.csv",
        "BCHUSDT": "bch_15m.csv",
    }
    fname = m.get(symbol)
    if not fname:
        return None
    resources_dir = resolve_resources_dir(cfg)
    path = os.path.join(resources_dir, fname)
    return path if os.path.exists(path) else None


def _coerce_float_or_zero(x):
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return 0.0
        return xf
    except Exception:
        return 0.0


def _weight(h: int) -> float:
    return math.sqrt(max(1, int(h)))


def _clean_prob_map(prob_map: dict) -> dict:
    clean = {}
    for k, v in (prob_map or {}).items():
        try:
            f = float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
        except Exception:
            continue
        if f < 0.0 or f > 1.0:
            continue
        clean[k] = f
    return clean


def pick_latest_valid_row(features_df: pd.DataFrame, k: int = 3):
    tail = features_df.tail(k)
    logging.info(f"[DIAG] tail_na_counts: {tail.isna().sum().to_dict()}")
    for idx in tail.index[::-1]:
        row = tail.loc[idx]
        if not row.isna().any() and np.isfinite(row.to_numpy(dtype=float)).all():
            return idx, row
    return None, None


def aggregate_signal(prob_map: dict, enter_threshold: float = 0.75, method: str = "max_weighted") -> dict:
    clean = _clean_prob_map(prob_map)
    if not clean:
        return {
            "side": "NONE",
            "score": 0.0,
            "prob_up_max": 0.0,
            "prob_down_max": 1.0,
            "chosen_h": None,
            "chosen_t": None,
            "reason": "empty_or_nan",
        }

    if method == "majority":
        ups = sum(1 for _,p in clean.items() if p >= enter_threshold)
        downs = sum(1 for _,p in clean.items() if (1.0 - p) >= enter_threshold)
        if   ups > downs and ups > 0: side, score = "LONG", 1.0
        elif downs > ups and downs>0: side, score = "SHORT", 1.0
        else:                         side, score = "NONE", 0.0
        pu = max(clean.values())
        return {"side":side,"score":_coerce_float_or_zero(score),"prob_up_max":float(pu),
                "prob_down_max":float(1.0-pu),"chosen_h":None,"chosen_t":None,
                "reason":"majority"}

    scored = []
    total_weight = 0.0
    for (h,t), p in clean.items():
        w = _weight(h)
        total_weight += w
        scored.append(((h,t), p*w))
    if not scored or total_weight <= 0 or all(
        (s is None or math.isnan(s[1]) or s[1] == 0.0) for s in scored
    ):
        return {
            "side": "NONE",
            "score": 0.0,
            "prob_up_max": 0.0,
            "prob_down_max": 1.0,
            "chosen_h": None,
            "chosen_t": None,
            "reason": "empty_or_nan",
        }
    (chosen_ht, score) = max(scored, key=lambda x: x[1])
    (ch, ct) = chosen_ht
    pu = max(clean.values())
    side = "LONG" if pu >= enter_threshold else "NONE"
    score = _coerce_float_or_zero(score)
    return {"side":side,"score":score,"prob_up_max":float(pu),"prob_down_max":float(1.0-pu),
            "chosen_h":int(ch),"chosen_t":float(ct),
            "reason":"ok" if side!="NONE" else "below_threshold"}




def read_or_fetch_latest(
    symbol: str,
    *,
    cfg: Dict[str, Any],
    interval: str = "15m",
    now_ts: Optional[pd.Timestamp] = None,
    limit: int = 210,
):
    try:
        fetch_fn = _select_fetcher(cfg)
        if fetch_fn is None:
            log_diag(f"{symbol}: fetch disabled (csv_only/none) -> use local CSV only")
        elif not callable(fetch_fn):
            log_diag(
                f"{symbol}: BAD_FETCH_FN type={type(fetch_fn)} repr={repr(fetch_fn)}"
            )
            return {"side": "NONE", "score": 0.0, "reason": "bad_fetch_fn"}

        csv_path = _csv_path_for_symbol(symbol, cfg)
        if not csv_path:
            log_diag(f"{symbol}: csv not found under resources_dir -> stale_data")
            return {"side": "NONE", "score": 0.0, "reason": "stale_data"}

        interval_td = pd.to_timedelta(interval)
        log_diag(
            f"read_or_fetch_latest: now_ts_in={now_ts} (type={type(now_ts)})"
        )

        try:
            now_ts = safe_ts_to_utc(now_ts)
            log_diag(
                f"read_or_fetch_latest: now_ts_utc={now_ts} tz={getattr(now_ts,'tzinfo',None)}"
            )
        except Exception as e:
            log_trace("CONVERT_NOW_TS_FAIL", e)
            return {
                "side": "NONE",
                "score": 0.0,
                "reason": f"LOOP_EXCEPTION:{type(e).__name__}",
            }

        anchor = last_closed_15m(now_ts)

        path = Path(csv_path)
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        df = ensure_utc_index(df, "timestamp")
        logger.debug(
            "[DIAG] df.index.tz=%s, head_ts=%s, now_ts=%s, safe_ts_to_utc=%s",
            getattr(df.index.tz, 'key', df.index.tz),
            list(df.index[:3]),
            now_ts,
            type(safe_ts_to_utc).__name__,
        )
        assert str(df.index.tz) == "UTC", "[DIAG] index not UTC"

        latest_close = df.index.max() if not df.empty else pd.NaT
        lag = (anchor - latest_close) if pd.notna(latest_close) else pd.Timedelta.max
        is_stale = pd.isna(latest_close) or lag >= interval_td
        retried = 0

        if is_stale and callable(fetch_fn):
            retried = 1
            end_time = anchor
            start_time = end_time - interval_td * max(limit, 210)
            new_df = fetch_fn(
                symbol,
                interval,
                int(start_time.timestamp() * 1000),
                int(end_time.timestamp() * 1000),
            )
            df = pd.concat([df, new_df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            safe_reset_index(df, name="timestamp", overwrite=True).to_csv(
                path, index=False
            )
            latest_close = df.index.max() if not df.empty else pd.NaT
            lag = (anchor - latest_close) if pd.notna(latest_close) else pd.Timedelta.max
            is_stale = pd.isna(latest_close) or lag >= interval_td

        diff_min = lag.total_seconds() / 60 if pd.notna(latest_close) else float("inf")
        print(
            f"[TIME] anchor={anchor.isoformat()} latest_close={latest_close.isoformat() if pd.notna(latest_close) else 'none'} diff_min={diff_min:.2f}"
        )
        print(f"[FETCH] retried={retried} endTime={anchor.isoformat()}")
        return df
    except Exception as e:
        log_trace("LOOP_EXCEPTION(read_or_fetch_latest)", e)
        return {"side": "NONE", "score": 0.0, "reason": f"LOOP_EXCEPTION:{type(e).__name__}"}


# get_latest_signal（如已存在，請覆蓋為更嚴格版）
def get_latest_signal(
    symbol: str,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    models: Dict[str, Any],
    now_ts: Optional[pd.Timestamp] = None,
):
    try:
        require_models = bool((cfg or {}).get("require_models", True))
        if require_models and not models:
            log_diag(f"{symbol}: no models loaded but require_models=True")
            return {"side": "NONE", "score": 0.0, "reason": "no_models_loaded"}

        rt = (cfg or {}).get("realtime", {})
        stale_thr = float(rt.get("stale_threshold_min", 30))

        latest_close = ensure_aware_utc(df.index.max())
        anchor = (
            ensure_aware_utc(safe_ts_to_utc(now_ts))
            if now_ts is not None
            else latest_close
        )
        if latest_close > anchor:
            anchor = latest_close
        lag_minutes = (anchor - latest_close).total_seconds() / 60.0
        log_diag(
            f"{symbol}: latest_ts={latest_close} anchor={anchor} diff_min={lag_minutes:.2f}"
        )

        now_actual = ensure_aware_utc(safe_ts_to_utc(None))
        if (now_actual - latest_close).total_seconds() / 60.0 > stale_thr:
            log_diag(f"{symbol}: stale by >{stale_thr} min (latest={latest_close})")
            return {"side": "NONE", "score": 0.0, "reason": "stale_data"}

        score = 0.0
        side = "NONE"
        reason = "ok"

        return {"side": side, "score": float(score), "reason": reason}

    except Exception as e:
        log_trace("LOOP_EXCEPTION(get_latest_signal)", e)
        return {"side": "NONE", "score": 0.0, "reason": f"LOOP_EXCEPTION:{type(e).__name__}"}
