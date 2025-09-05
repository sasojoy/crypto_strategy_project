from __future__ import annotations

import sys, math, traceback, os, json, logging
from typing import Optional
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


TZ_TW = tz.gettz("Asia/Taipei")
logger = logging.getLogger(__name__)


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
    csv_path: str,
    *,
    interval: str = "15m",
    now_ts: Optional[pd.Timestamp] = None,
    limit: int = 210,
):
    try:
        interval_td = pd.to_timedelta(interval)
        log_diag(
            f"callable(safe_ts_to_utc)={callable(safe_ts_to_utc)} type={type(safe_ts_to_utc)}"
        )
        log_diag(
            f"read_or_fetch_latest: now_ts_in={now_ts} (type={type(now_ts)})"
        )
        for _name in ("fetch_fn", "fetch_latest", "loader"):
            if _name in locals():
                _val = locals()[_name]
                if not callable(_val):
                    log_diag(
                        f"BAD_FETCH_FN name={_name} repr={repr(_val)} type={type(_val)}"
                    )
                    return {"side": "NONE", "score": 0.0, "reason": "bad_fetch_fn"}
        if not callable(fetch_klines_range):
            log_diag(
                f"BAD_FETCH_FN name=fetch_klines_range repr={repr(fetch_klines_range)} type={type(fetch_klines_range)}"
            )
            return {"side": "NONE", "score": 0.0, "reason": "bad_fetch_fn"}
        try:
            now_ts = safe_ts_to_utc(now_ts)
            log_diag(
                f"read_or_fetch_latest: now_ts_utc={now_ts} tz={getattr(now_ts,'tzinfo',None)}"
            )
        except Exception as e:
            log_trace("CONVERT_NOW_TS_FAIL", e)
            return {"side": "NONE", "score": 0.0, "reason": f"LOOP_EXCEPTION:{type(e).__name__}"}

        if 'start_ts' in locals() and start_ts is not None:
            start_ts = safe_ts_to_utc(start_ts)
        if 'end_ts' in locals() and end_ts is not None:
            end_ts = safe_ts_to_utc(end_ts)

        anchor = last_closed_15m(now_ts)

        path = Path(csv_path)
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
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

        if is_stale:
            retried = 1
            end_time = anchor
            start_time = end_time - interval_td * max(limit, 210)
            new_df = fetch_klines_range(
                symbol,
                interval,
                int(start_time.timestamp() * 1000),
                int(end_time.timestamp() * 1000),
            )
            df = pd.concat([df, new_df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.reset_index().to_csv(path, index=False)
            latest_close = df.index.max() if not df.empty else pd.NaT
            lag = (anchor - latest_close) if pd.notna(latest_close) else pd.Timedelta.max
            is_stale = pd.isna(latest_close) or lag >= interval_td

        diff_min = lag.total_seconds() / 60 if pd.notna(latest_close) else float("inf")
        print(
            f"[TIME] anchor={anchor.isoformat()} latest_close={latest_close.isoformat() if pd.notna(latest_close) else 'none'} diff_min={diff_min:.2f}"
        )
        print(f"[FETCH] retried={retried} endTime={anchor.isoformat()}")
        return df, anchor, latest_close, is_stale
    except Exception as e:
        log_trace("LOOP_EXCEPTION(read_or_fetch_latest)", e)
        return {"side": "NONE", "score": 0.0, "reason": f"LOOP_EXCEPTION:{type(e).__name__}"}


# get_latest_signal（如已存在，請覆蓋為更嚴格版）
def get_latest_signal(symbol: str, cfg: dict, fresh_min: float = 5.0, *, debug: bool = False) -> dict | None:
    from csp.core.feature import add_features
    try:
        from csp.models.classifier_multi import MultiThresholdClassifier
    except Exception:
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}

    try:
        csv_path = cfg["io"]["csv_paths"].get(symbol)
        if not csv_path or not os.path.exists(csv_path):
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_data"}

        res = read_or_fetch_latest(symbol, csv_path)
        if isinstance(res, dict):
            return res
        df, anchor, latest_close, is_stale = res
        try:
            latest_close = ensure_aware_utc(latest_close)
            anchor = ensure_aware_utc(anchor)
            lag_minutes = (anchor - latest_close).total_seconds() / 60.0
            log_diag(
                f"latest_ts={latest_close} anchor={anchor} diff_min={lag_minutes:.2f}"
            )
        except Exception as e:
            log_trace("LAG_CALC_FAIL", e)
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": f"LOOP_EXCEPTION:{type(e).__name__}"}
        if is_stale:
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "stale_data"}
        if lag_minutes > fresh_min:
            return None

        dff = add_features(df.copy())

        model_dir = os.path.join(cfg["io"].get("models_dir", "models"), symbol, "cls_multi")
        if debug:
            files = os.listdir(model_dir) if os.path.exists(model_dir) else []
            log_diag(f"model_dir={model_dir} files={files} total={len(files)}")
        meta_path = os.path.join(model_dir, "meta.json")
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}
        if not os.path.exists(meta_path):
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        feat_cols = meta.get("feature_columns") or []
        if not feat_cols:
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}
        x = dff.iloc[[-1]].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        x = x.reindex(feat_cols, axis=1)
        missing = [c for c in feat_cols if c not in dff.columns]
        if missing:
            log_diag(f"feature_missing={missing}")
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "feature_mismatch"}
        nan_cols = x.columns[x.iloc[0].isna()].tolist()
        if nan_cols:
            log_diag(f"feature_nan_cols={nan_cols}")
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "feature_nan"}

        m = MultiThresholdClassifier.load(model_dir)
        model_files = os.listdir(model_dir)
        log_diag(f"models_loaded={model_files}")
        prob_map = m.predict_proba(x)
        log_diag(f"predict_proba={prob_map}")
        prob_map = {k: float(v) for k, v in prob_map.items() if v is not None and not pd.isna(v)}
        if not prob_map:
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "empty_or_invalid_inputs"}
        th = cfg.get("strategy", {}).get("enter_threshold", 0.75)
        method = cfg.get("strategy", {}).get("aggregator_method", "max_weighted")
        sig = aggregate_signal(prob_map, enter_threshold=th, method=method)
        if not sig.get("side"):
            sig["side"] = "NONE"
        sig["score"] = _coerce_float_or_zero(sig.get("score"))
        price = float(df["close"].iloc[-1]) if not df.empty else 0.0
        sig["price"] = price
        sig["symbol"] = symbol
        sig["ts"] = now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")
        log_diag(
            f"final side={sig['side']} score={sig.get('score')} reason={sig.get('reason')}"
        )
        return sig
    except Exception as e:
        log_trace("LOOP_EXCEPTION(get_latest_signal)", e)
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": f"LOOP_EXCEPTION:{type(e).__name__}"}
