from __future__ import annotations
import math, os, json, logging
import pandas as pd
from dateutil import tz
from typing import Optional

from pathlib import Path
import numpy as np
from csp.data.binance import fetch_klines_range
from csp.utils.timez import ensure_utc_index, last_closed_15m
# 改為以模組命名空間導入，避免函式名被區域/參數名遮蔽
from csp.utils import time as time_utils


TZ_TW = tz.gettz("Asia/Taipei")
logger = logging.getLogger(__name__)

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
        return {"side":side,"score":float(score),"prob_up_max":float(pu),
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
    score = 0.0 if (score is None or math.isnan(score)) else float(score)
    return {"side":side,"score":score,"prob_up_max":float(pu),"prob_down_max":float(1.0-pu),
            "chosen_h":int(ch),"chosen_t":float(ct),
            "reason":"ok" if side!="NONE" else "below_threshold"}


def sanitize_score(x):
    import math
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def read_or_fetch_latest(
    symbol: str,
    csv_path: str,
    *,
    interval: str = "15m",
    now_ts: Optional[pd.Timestamp] = None,
    limit: int = 210,
):
    interval_td = pd.to_timedelta(interval)
    # 防呆：確保工具函式沒有被遮蔽
    assert callable(time_utils.now_utc), "now_utc not callable (possibly shadowed)"
    assert callable(time_utils.safe_ts_to_utc), "safe_ts_to_utc not callable (possibly shadowed)"

    if now_ts is None:
        now_ts = time_utils.now_utc()
    else:
        now_ts = time_utils.safe_ts_to_utc(now_ts)
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
        type(time_utils.safe_ts_to_utc).__name__,
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


# get_latest_signal（如已存在，請覆蓋為更嚴格版）
def get_latest_signal(symbol: str, cfg: dict, fresh_min: float = 5.0, *, debug: bool = False) -> dict | None:
    from csp.core.feature import add_features
    try:
        from csp.models.classifier_multi import MultiThresholdClassifier
    except Exception:
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}

    csv_path = cfg["io"]["csv_paths"].get(symbol)
    if not csv_path or not os.path.exists(csv_path):
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_data"}

    df, anchor, latest_close, is_stale = read_or_fetch_latest(symbol, csv_path)
    lag_minutes = (anchor - latest_close).total_seconds() / 60 if pd.notna(latest_close) else float("inf")
    print(f"[DIAG] latest_ts={latest_close} anchor={anchor} diff_min={lag_minutes:.2f}")
    if is_stale:
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "stale_data"}
    if lag_minutes > fresh_min:
        return None

    dff = add_features(df.copy())

    model_dir = os.path.join(cfg["io"].get("models_dir", "models"), symbol, "cls_multi")
    if debug:
        files = os.listdir(model_dir) if os.path.exists(model_dir) else []
        print(f"[DEBUG] model_dir={model_dir} files={files} total={len(files)}")
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}
    if not os.path.exists(meta_path):
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    feat_cols = meta.get("feature_columns") or []
    if not feat_cols:
        return {"symbol": symbol, "side": "NONE", "score": None, "reason": "no_models_loaded"}
    x = dff.iloc[[-1]].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    x = x.reindex(feat_cols, axis=1)
    missing = [c for c in feat_cols if c not in dff.columns]
    if missing:
        print(f"[DIAG] feature_missing={missing}")
        return {"symbol": symbol, "side": "NONE", "score": None, "reason": "feature_mismatch"}
    nan_cols = x.columns[x.iloc[0].isna()].tolist()
    if nan_cols:
        print(f"[DIAG] feature_nan_cols={nan_cols}")
        return {"symbol": symbol, "side": "NONE", "score": None, "reason": "feature_nan"}

    m = MultiThresholdClassifier.load(model_dir)
    model_files = os.listdir(model_dir)
    print(f"[DIAG] models_loaded={model_files}")
    prob_map = m.predict_proba(x)
    print(f"[DIAG] predict_proba={prob_map}")
    prob_map = {k: float(v) for k, v in prob_map.items() if v is not None and not pd.isna(v)}
    if not prob_map:
        return {"symbol": symbol, "side": "NONE", "score": None, "reason": "empty_or_invalid_inputs"}
    th = cfg.get("strategy", {}).get("enter_threshold", 0.75)
    method = cfg.get("strategy", {}).get("aggregator_method", "max_weighted")
    sig = aggregate_signal(prob_map, enter_threshold=th, method=method)
    if not sig.get("side"):
        sig["side"] = "NONE"
    sig["score"] = sanitize_score(sig.get("score"))
    price = float(df["close"].iloc[-1]) if not df.empty else 0.0
    sig["price"] = price
    sig["symbol"] = symbol
    sig["ts"] = time_utils.now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[DIAG] final side={sig['side']} score={sig.get('score')} reason={sig.get('reason')}")
    return sig
