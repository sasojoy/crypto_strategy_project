from __future__ import annotations
import math, os, json
import pandas as pd
from dateutil import tz

from pathlib import Path
import numpy as np
from csp.data.binance import fetch_klines_range
from csp.utils.tz_safe import (
    normalize_df_to_utc,
    safe_ts_to_utc,
    now_utc,
    floor_utc,
)


TZ_TW = tz.gettz("Asia/Taipei")

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
        return 0.0
    if isinstance(x, float) and math.isnan(x):
        return 0.0
    try:
        return float(x)
    except Exception:
        return 0.0


def read_or_fetch_latest(
    symbol: str,
    csv_path: str,
    *,
    interval: str = "15m",
    now_utc: pd.Timestamp | None = None,
    limit: int = 210,
):
    interval_td = pd.to_timedelta(interval)
    now_ts = now_utc() if now_utc is None else safe_ts_to_utc(now_utc)
    floor_now = floor_utc(now_ts, interval)

    path = Path(csv_path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = normalize_df_to_utc(df)
    print(f"[DIAG] df.index.tz={df.index.tz}, head_ts={df.index[:3].tolist()}")
    assert str(df.index.tz) == "UTC", "[DIAG] index not UTC"

    latest_close = df.index[-1] if not df.empty else pd.NaT
    latest_open = latest_close - interval_td if pd.notna(latest_close) else pd.NaT
    bar_close_exact = latest_close
    match = bool(bar_close_exact == floor_now)
    is_stale = pd.isna(bar_close_exact) or bar_close_exact < floor_now
    retried = 0

    if is_stale:
        retried = 1
        end_time = floor_now
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
        latest_close = df.index[-1] if not df.empty else pd.NaT
        latest_open = latest_close - interval_td if pd.notna(latest_close) else pd.NaT
        bar_close_exact = latest_close
        match = bool(bar_close_exact == floor_now)
        is_stale = pd.isna(bar_close_exact) or bar_close_exact < floor_now

    print(
        f"[TIME] now_utc={now_ts.isoformat()} floor_now={floor_now.isoformat()} "
        f"latest_open={latest_open.isoformat() if pd.notna(latest_open) else 'none'} "
        f"latest_close={latest_close.isoformat() if pd.notna(latest_close) else 'none'} "
        f"bar_close_exact={bar_close_exact.isoformat() if pd.notna(bar_close_exact) else 'none'} "
        f"match={match}"
    )
    print(f"[FETCH] retried={retried} endTime={floor_now.isoformat()}")
    return df, floor_now, bar_close_exact, is_stale


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

    df, floor_now, bar_close_exact, is_stale = read_or_fetch_latest(symbol, csv_path)
    if is_stale:
        return {
            "symbol": symbol,
            "side": "NONE",
            "score": 0.0,
            "reason": "stale_data_after_refresh",
        }

    now_ts = now_utc()
    lag_minutes = (now_ts - bar_close_exact).total_seconds() / 60.0
    if lag_minutes > 15:
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "stale_data"}
    if lag_minutes > fresh_min:
        return None

    dff = add_features(df.copy())

    model_dir = os.path.join(cfg["io"].get("models_dir", "models"), symbol, "cls_multi")
    if debug:
        files = os.listdir(model_dir) if os.path.exists(model_dir) else []
        print(f"[DEBUG] model_dir={model_dir} files={files} total={len(files)}")
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        print("[MODELS] loaded=0")
        print("[FEATURE] last_row_nan_cols=[]")
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    feat_cols = meta.get("feature_columns") or []
    if not feat_cols:
        print("[MODELS] loaded=0")
        print("[FEATURE] last_row_nan_cols=[]")
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}
    for c in feat_cols:
        if c not in dff.columns:
            print("[FEATURE] last_row_nan_cols=[]")
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "feature_missing"}

    X = dff[feat_cols].tail(1)
    nan_cols = X.columns[X.isna().any()].tolist()
    print(f"[FEATURE] last_row_nan_cols={nan_cols}")
    if nan_cols:
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "feature_nan"}

    m = MultiThresholdClassifier.load(model_dir)
    print(f"[MODELS] loaded={len(m.models)}")
    prob_map = m.predict_proba(X)
    th = cfg.get("strategy", {}).get("enter_threshold", 0.75)
    method = cfg.get("strategy", {}).get("aggregator_method", "max_weighted")
    sig = aggregate_signal(prob_map, enter_threshold=th, method=method)
    if not sig.get("side"):
        sig["side"] = "NONE"
    sig["score"] = sanitize_score(sig.get("score"))
    price = float(df["close"].iloc[-1]) if not df.empty else 0.0
    sig["price"] = price
    sig["symbol"] = symbol
    sig["ts"] = now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")
    return sig
