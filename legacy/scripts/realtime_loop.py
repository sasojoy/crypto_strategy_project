from __future__ import annotations

import argparse
import json
import time
import os
import sys
import math
import traceback
import socket
from dataclasses import asdict

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dateutil import tz

from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import hashlib, inspect
import csp
from csp import notify as tg_notify
from csp.utils.notifier import (
    notify_signal,
    notify_trade_open,
    notify_guard,
)
from csp.utils.bar_time import align_15m
from csp.utils.signal_context import build_signal_context

try:
    # 供 min_notional 檢查（若你之後移檔，這裡記得同步 import 路徑）
    from csp.exchange.minnotional import get_min_notional
except Exception:
    # 若模組尚未存在，延後在執行到下單 Guard 時再報錯，避免影響 signal_only
    get_min_notional = None  # type: ignore

from csp.utils.diag import log_diag, log_trace


def _install_global_excepthook():
    def _hook(exc_type, exc, tb):
        try:
            log_trace("GLOBAL_EXCEPTION", exc)
        except Exception:
            pass

    sys.excepthook = _hook


_install_global_excepthook()

if os.environ.get("DIAG_SELFTEST") == "1":
    try:
        raise RuntimeError("DIAG_SELFTEST_TRIGGER")
    except Exception as e:
        log_trace("SELFTEST", e)

_APP_FILE = Path(__file__).resolve()
_APP_SHA8 = hashlib.sha1(_APP_FILE.read_bytes()).hexdigest()[:8]
try:
    _REL = json.loads((Path(__file__).resolve().parents[1] / "RELEASE.json").read_text())
    _BUILD = (_REL.get("sha") or _APP_SHA8)[:8]
    _BRANCH = _REL.get("branch") or "unknown"
    _BUILT_AT = _REL.get("built_at") or "unknown"
except Exception:
    _BUILD, _BRANCH, _BUILT_AT = _APP_SHA8, "unknown", "unknown"
print(f"[BOOT] realtime_loop.py={_APP_FILE}")
print(f"[BOOT] file_sha8={_APP_SHA8}  build={_BUILD}  branch={_BRANCH}  built_at={_BUILT_AT}")
print(f"[BOOT] csp_module_path={inspect.getfile(csp)}")

from csp.strategy.model_hub import load_models
from csp.utils.paths import resolve_resources_dir
from csp.utils.timez import now_utc, UTC, ensure_utc_index
from csp.strategy.position_sizing import (
    blended_sizing, SizingInput, ExchangeRule, kelly_fraction
)
from csp.runtime.exit_watchdog import check_exit_once
from csp.utils.io import load_cfg
from csp.data.public_klines import fetch_binance_klines_public


def ensure_latest_closed_anchor(now_utc: "pd.Timestamp") -> "pd.Timestamp":
    # 回傳「當前應該使用的 15m 收盤時間點」（不含尚未收盤的 bar）
    floored = now_utc.floor("15min")
    # 若剛好是整點（:00/:15/:30/:45），上一根才是最後已收盤
    if now_utc == floored:
        return floored - pd.Timedelta(minutes=15)
    return floored


def ensure_not_stale(df: "pd.DataFrame", symbol: str, now_utc: "pd.Timestamp") -> Tuple[bool, str]:
    """檢查 df 是否有覆蓋到最新已收盤 anchor，一旦落後就回傳 (False, reason)。"""
    if df.empty:
        return False, "empty_df"
    df = ensure_utc_index(df)
    last_close = df.index.max()
    anchor = ensure_latest_closed_anchor(now_utc)
    diff_min = (anchor - last_close).total_seconds() / 60.0
    if diff_min > 0:
        return False, f"stale_data(anchor={anchor.isoformat()}, latest_close={last_close.isoformat()}, diff_min={diff_min:.2f})"
    return True, ""


def load_feature_names(models_dir: str, symbol: str) -> List[str]:
    # 僅允許由模型輸出的特徵清單，推論必須以它為準
    p = os.path.join(models_dir, symbol, "feature_names.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"feature_names.json not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError(f"feature_names.json invalid/empty for {symbol}")
    return names


def validate_numeric_features(feats: "pd.DataFrame", cols: List[str]) -> None:
    bad = []
    for c in cols:
        if c not in feats.columns:
            bad.append(f"{c} (missing)")
        else:
            if not pd.api.types.is_numeric_dtype(feats[c].dtype):
                bad.append(f"{c} ({feats[c].dtype})")
    if bad:
        raise ValueError("Non-numeric or missing feature columns detected: " + ", ".join(bad))


def align_features_for_infer(
    feats_reset: "pd.DataFrame",
    models_dir: str,
    symbol: str,
    scaler,
) -> "pd.DataFrame":
    """
    1) 讀取模型的 feature_names.json 作為唯一可信來源與順序
    2) 排除非數值欄位（若 feature_names.json 竟含 timestamp，直接報錯）
    3) 若 scaler.n_features_in_ 存在，強制長度一致；不一致就回傳清楚錯誤，請重訓或修正 metadata
    """
    names = load_feature_names(models_dir, symbol)
    forbidden = [c for c in names if c.lower() in ("timestamp", "ts", "time")]
    if forbidden:
        raise ValueError(
            f"feature_names.json for {symbol} contains non-numeric time-like columns: {forbidden}. Please retrain/export metadata correctly."
        )
    X = feats_reset.loc[:, names].copy()
    validate_numeric_features(feats_reset, names)
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        expected = int(scaler.n_features_in_)
        if X.shape[1] != expected:
            msg = (
                f"[FEATURE_MISMATCH] {symbol}: X has {X.shape[1]} features, but scaler expects {expected}.\n"
                f"names_from_model={len(names)}; first10={names[:10]}"
            )
            raise ValueError(msg)
    return X


def sanitize_score(x):
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return 0.0
        return xf
    except Exception:
        return 0.0


def _extract_threshold_value(data: Any, horizon: int) -> Optional[float]:
    if isinstance(data, (int, float)) and not math.isnan(float(data)):
        return float(data)
    if isinstance(data, dict):
        key = str(horizon)
        if key in data:
            val = data[key]
            if isinstance(val, dict):
                for inner_key in ("threshold", "thr", "value", "default"):
                    inner_val = val.get(inner_key)
                    if isinstance(inner_val, (int, float)):
                        return float(inner_val)
            elif isinstance(val, (int, float)):
                return float(val)
        for cand in ("threshold", "default_threshold", "default", "long", "value"):
            val = data.get(cand)
            if isinstance(val, (int, float)):
                return float(val)
        for val in data.values():
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, dict):
                nested = _extract_threshold_value(val, horizon)
                if nested is not None:
                    return nested
    return None


def load_thresholds_by_symbol(cfg: Dict[str, Any]) -> Dict[str, float]:
    models_dir = Path(cfg.get("io", {}).get("models_dir", "models"))
    horizon = int(cfg.get("train", {}).get("target_horizon_bars", 16))
    thresholds: Dict[str, float] = {}

    global_file = models_dir / "thresholds.json"
    if global_file.exists():
        try:
            data = json.loads(global_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for sym, val in data.items():
                    thr = _extract_threshold_value(val, horizon)
                    if thr is not None:
                        thresholds[sym.upper()] = thr
        except Exception:
            pass

    for sym in cfg.get("symbols", []):
        sym_dir = models_dir / sym
        for fname in ("thresholds.json", "threshold.json"):
            f = sym_dir / fname
            if not f.exists():
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                thr = _extract_threshold_value(data, horizon)
                if thr is not None:
                    thresholds[sym.upper()] = thr
            except Exception:
                continue
    return thresholds


def compute_atr_from_history(df: pd.DataFrame, n: int) -> Optional[float]:
    if df is None or df.empty:
        return None
    if len(df) < max(n, 2):
        return None
    try:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
    except Exception:
        return None
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_series = tr.rolling(int(n)).mean()
    atr_value = atr_series.iloc[-1]
    if atr_value is None or math.isnan(float(atr_value)):
        return None
    return float(atr_value)


def _state_signals(state: Dict[str, Any]) -> Dict[str, Any]:
    sigs = state.get("signals")
    if isinstance(sigs, dict):
        return sigs
    sigs = {}
    state["signals"] = sigs
    return sigs


def cooldown_ok(symbol: str, bar_open_ts: pd.Timestamp, state: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    cooldown_bars = int(cfg.get("entry_filter", {}).get("reentry_cooldown_bars", 0))
    if cooldown_bars <= 0:
        return True
    sigs = _state_signals(state)
    last = sigs.get(symbol, {}).get("bar_open")
    if not last:
        return True
    try:
        last_ts = pd.Timestamp(last)
    except Exception:
        return True
    diff = (bar_open_ts - last_ts) / pd.Timedelta(minutes=15)
    try:
        diff_val = float(diff)
    except Exception:
        return True
    return diff_val >= cooldown_bars


def current_drawdown_ok(symbol: str, state: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    # Placeholder: drawdown guard not implemented yet, always pass.
    return True


def session_ok(now_utc: pd.Timestamp, cfg: Dict[str, Any]) -> bool:
    # Placeholder for session guard; allow all sessions by default.
    return True


def vol_ok(atr_value: Optional[float], cfg: Dict[str, Any]) -> bool:
    # Basic volatility guard: if max_atr_pct configured, ensure atr/price within range.
    max_pct = cfg.get("risk", {}).get("max_atr_pct")
    if max_pct is None or atr_value is None:
        return True
    try:
        return float(atr_value) <= float(max_pct)
    except Exception:
        return True

TW = tz.gettz("Asia/Taipei")
logger = logging.getLogger(__name__)


STATE_FILE = Path("/tmp/realtime_state.json")


def _load_dispatch_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[WARN] failed to read state file %s: %s", STATE_FILE, exc)
        return {}


def _save_dispatch_state(state: Dict[str, Any]) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("[WARN] failed to write state file %s: %s", STATE_FILE, exc)




def pick_latest_valid_row(df_feat: "pd.DataFrame", k: int = 3):
    # 僅針對 numeric 欄位檢查有限值
    numeric_df = df_feat.select_dtypes(include=[np.number])
    for idx in range(len(df_feat) - 1, -1, -1):
        row_num = numeric_df.iloc[idx]
        if not row_num.isna().any() and np.isfinite(row_num.to_numpy(dtype=float)).all():
            return idx, df_feat.iloc[idx]
    raise RuntimeError("No valid feature row found in last k bars")


# ========= 通知摘要格式 =========
def _summarize_proba(d: Dict[str, Any]) -> str:
    """把 {'2':0.61,'16':0.74} 這類字典，轉成簡短字串 h2=0.61,h16=0.74（最多 4 個）。"""
    if not d:
        return "-"
    items = list(d.items())
    try:
        items.sort(key=lambda kv: int(kv[0]))
    except Exception:
        pass
    items = items[:4]
    return ", ".join([f"h{str(k)}={float(v):.2f}" for k, v in items])


def format_signal_summary(one: Dict[str, Any]) -> str:
    """單幣別摘要行：BTCUSDT: LONG | 0.957 | chosen_h=16 | proba=h2=0.61,h16=0.74"""
    sym = one.get("symbol")
    side = one.get("side", "NONE")
    score = one.get("score", 0.0)
    ch = one.get("chosen_h") or "-"
    pb = _summarize_proba(one.get("proba_by_h") or {})
    rsn = one.get("reason") or "-"
    return f"{sym}: {side} | score={score:.3f} | chosen_h={ch} | proba={pb} | reason={rsn}"



def predict_one(symbol: str, df_15m: pd.DataFrame, model, scaler, cfg_path: str = "csp/configs/strategy.yaml"):
    """Build features, pick latest valid row and run prediction."""
    from csp.features.h16 import build_features_15m_4h
    from csp.core.feature import add_features
    from csp.utils.config import get_symbol_features

    cfg = load_cfg(cfg_path)
    feat_params = get_symbol_features(cfg, symbol)

    df_15m = ensure_utc_index(df_15m)
    feats = build_features_15m_4h(
        df_15m,
        ema_windows=tuple(feat_params["ema_windows"]),
        rsi_window=feat_params["rsi_window"],
        bb_window=feat_params["bb_window"],
        bb_std=feat_params["bb_std"],
        atr_window=feat_params["atr_window"],
        h4_resample=feat_params["h4_resample"],
    )
    feats = add_features(
        feats,
        prev_high_period=feat_params["prev_high_period"],
        prev_low_period=feat_params["prev_low_period"],
        bb_window=feat_params["bb_window"],
        atr_window=feat_params["atr_window"],
        atr_percentile_window=feat_params["atr_percentile_window"],
    )

    feats = feats.reset_index()  # 這會把 timestamp 變成一個欄位
    models_dir = cfg.get("io", {}).get("models_dir", "models")
    X_all = align_features_for_infer(feats, models_dir, symbol, scaler)
    try:
        idx, _ = pick_latest_valid_row(X_all, k=3)
    except RuntimeError:
        logging.warning(
            f"[WARN] no valid feature row for {symbol} (last 3 bars all contain NaN)."
        )
        return {
            "symbol": symbol,
            "side": "NONE",
            "score": None,
            "reason": "no_valid_features",
        }

    x = X_all.iloc[[idx]]
    try:
        # 統一丟 numpy，避免 sklearn 「has feature names」警告
        x2 = scaler.transform(x.values) if scaler is not None else x.values
        proba = model.predict_proba(x2)[0, 1]
        score = float(proba)
    except Exception as e:
        logging.exception(f"[ERROR] predict failed for {symbol}: {e}")
        return {
            "symbol": symbol,
            "side": "NONE",
            "score": None,
            "reason": "predict_exception",
        }

    out = {"symbol": symbol, "score": score}
    if "proba_by_h" in locals():
        out["proba_by_h"] = proba_by_h
    if "chosen_h" in locals():
        out["chosen_h"] = chosen_h
    if "chosen_t" in locals():
        out["chosen_t"] = chosen_t
    return out


def floor_to_interval(ts: datetime, interval: str) -> datetime:
    if interval.endswith("m"):
        step = int(interval[:-1])
        minute = (ts.minute // step) * step
        return ts.replace(minute=minute, second=0, microsecond=0)
    if interval.endswith("h"):
        step = int(interval[:-1])
        hour = (ts.hour // step) * step
        return ts.replace(hour=hour, minute=0, second=0, microsecond=0)
    raise ValueError(f"Unsupported interval: {interval}")


def interval_to_timedelta(interval: str) -> timedelta:
    if interval.endswith("m"):
        return timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return timedelta(hours=int(interval[:-1]))
    raise ValueError(f"Unsupported interval: {interval}")


def read_or_fetch_latest(cfg, symbol: str, csv_path: str, now_ts_in=None):
    """Read local CSV; if stale and fetching enabled, backfill via public API."""
    fetch_cfg = cfg.get("fetch", {}) or {}
    fetch_mode = fetch_cfg.get("mode", "csv_only")
    interval = fetch_cfg.get("interval", "15m")
    api_base = fetch_cfg.get("base_url", fetch_cfg.get("api_base", "https://api.binance.com"))
    writeback_csv = bool(fetch_cfg.get("writeback_csv", True))
    max_backfill_minutes = int(fetch_cfg.get("max_backfill_minutes", 360))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop(columns=["timestamp"])
        df.index = ts
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = None

    now_ts_utc = now_utc() if now_ts_in is None else now_ts_in
    anchor = floor_to_interval(now_ts_utc, interval)
    latest_close = df.index.max() if not df.empty else None

    if fetch_mode in ("none", "csv_only"):
        logger.debug(
            "[DIAG] %s: fetch disabled (csv_only/none) -> use local CSV only", symbol
        )
        return df, latest_close

    if latest_close is None:
        start_fetch = anchor - timedelta(minutes=max_backfill_minutes)
    else:
        start_fetch = latest_close + interval_to_timedelta(interval)
    end_fetch = anchor

    if start_fetch < end_fetch:
        try:
            if fetch_mode == "public_binance":
                new_df = fetch_binance_klines_public(
                    symbol,
                    start_fetch,
                    end_fetch,
                    interval=interval,
                    api_base=api_base,
                )
            else:
                raise ValueError(f"Unknown fetch mode: {fetch_mode}")

            if not new_df.empty:
                combined = pd.concat([df, new_df], axis=0).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                if writeback_csv:
                    out = combined.copy()
                    out.insert(0, "timestamp", out.index)
                    out.to_csv(csv_path, index=False)
                df = combined
                latest_close = df.index.max()
                logger.info(
                    "[FETCH] %s added rows=%d new_latest=%s",
                    symbol,
                    len(new_df),
                    latest_close.isoformat(),
                )
            else:
                logger.info("[FETCH] %s no new rows", symbol)
        except Exception as e:
            logger.warning("[WARN] %s: fetch error: %s", symbol, e, exc_info=False)

    return df, latest_close

def process_symbol(
    symbol: str,
    cfg: dict,
    models: dict,
    csv_path: str,
    *,
    prev_open_ts: pd.Timestamp,
    bar_open_ts: pd.Timestamp,
    state: Dict[str, Any],
    thresholds_by_symbol: Dict[str, float],
    default_threshold: float,
    horizon_bars: int,
    now_utc_ts: pd.Timestamp,
):
    try:
        df = pd.read_csv(csv_path)
        df = ensure_utc_index(df)

        ok, reason = ensure_not_stale(df, symbol, bar_open_ts)
        if not ok:
            fetch_cfg = cfg.get("fetch", {}) or {}
            fetch_mode = fetch_cfg.get("mode", "csv_only")
            if fetch_mode not in ("none", "csv_only"):
                df, _ = read_or_fetch_latest(
                    cfg, symbol, csv_path, now_ts_in=bar_open_ts
                )
                df = ensure_utc_index(df)
                ok, reason = ensure_not_stale(df, symbol, bar_open_ts)

        if not ok:
            allow_one = cfg.get("realtime", {}).get("allow_stale_one_bar", True)
            if allow_one and "diff_min=15.00" in reason:
                logger.info(
                    "%s %s — allow_stale_one_bar=True, proceed with last_close",
                    symbol,
                    reason,
                )
            else:
                return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": reason}

        if prev_open_ts not in df.index:
            return {
                "symbol": symbol,
                "side": "NONE",
                "score": 0.0,
                "reason": "missing_prev_bar",
            }

        df_hist = df[df.index <= prev_open_ts].copy()
        if df_hist.empty:
            return {
                "symbol": symbol,
                "side": "NONE",
                "score": 0.0,
                "reason": "insufficient_history",
            }

        try:
            prev_close_px = float(df_hist.loc[prev_open_ts, "close"])
        except Exception:
            prev_close_px = float(df_hist.iloc[-1]["close"]) if not df_hist.empty else None

        curr_open_px: Optional[float]
        price_source = "bar_open"
        if bar_open_ts in df.index:
            curr_row = df.loc[bar_open_ts]
            if isinstance(curr_row, pd.DataFrame):
                curr_row = curr_row.iloc[-1]
            curr_open_px = float(curr_row["open"])
        else:
            fallback_px = float(df.iloc[-1]["close"]) if not df.empty else None
            curr_open_px = fallback_px
            price_source = "fallback_last_close"
            if fallback_px is not None:
                logger.warning(
                    "%s missing bar %s open — fallback last close=%.4f",
                    symbol,
                    bar_open_ts.isoformat(),
                    fallback_px,
                )
            else:
                logger.warning(
                    "%s missing bar %s open — no fallback price available",
                    symbol,
                    bar_open_ts.isoformat(),
                )

        if curr_open_px is None:
            return {
                "symbol": symbol,
                "side": "NONE",
                "score": 0.0,
                "reason": "no_price_available",
            }

        bundle = models.get(symbol)
        if not bundle:
            return {
                "symbol": symbol,
                "side": "NONE",
                "score": 0.0,
                "reason": "no_models_loaded",
            }
        model = bundle.get("model") if isinstance(bundle, dict) else bundle
        scaler = bundle.get("scaler") if isinstance(bundle, dict) else None
        sig = predict_one(
            symbol,
            df_hist,
            model,
            scaler,
            cfg_path="csp/configs/strategy.yaml",
        )

        raw_score = sig.get("score")
        score = sanitize_score(raw_score)
        atr_n = int(cfg.get("risk", {}).get("atr_n", 14))
        atr_value = compute_atr_from_history(df_hist, atr_n)
        filters = {
            "cooldown_pass": cooldown_ok(symbol, bar_open_ts, state, cfg),
            "dd_guard_pass": current_drawdown_ok(symbol, state, cfg),
            "session_pass": session_ok(now_utc_ts, cfg),
            "vol_pass": vol_ok(atr_value, cfg),
            "extra_reasons": [],
        }
        if atr_value is None:
            filters["extra_reasons"].append("atr=fallback")
        sig_reason = sig.get("reason")
        if sig_reason and sig_reason not in ("-", "OK"):
            filters["extra_reasons"].append(f"model={sig_reason}")

        threshold = thresholds_by_symbol.get(symbol, default_threshold)
        ctx = build_signal_context(
            symbol=symbol,
            score=score,
            entry_price=curr_open_px,
            horizon_bars=horizon_bars,
            threshold=float(threshold),
            atr_value=atr_value,
            filters=filters,
            cfg=cfg,
        )
        ctx_dict = asdict(ctx)
        prob_down = max(0.0, min(1.0, 1.0 - ctx.score))

        result = {
            "symbol": symbol,
            "side": ctx.side,
            "score": ctx.score,
            "threshold": float(threshold),
            "price": ctx.entry_price,
            "prev_close": prev_close_px,
            "price_source": price_source,
            "bar_open_ts": bar_open_ts,
            "prev_open_ts": prev_open_ts,
            "ts": bar_open_ts.isoformat(),
            "horizon_bars": ctx.h_bars,
            "pt_ratio": ctx.pt,
            "sl_ratio": ctx.sl,
            "up_price": ctx.up_price,
            "down_price": ctx.down_price,
            "prob_up_max": ctx.score,
            "prob_down_max": prob_down,
            "reason": ctx.reason,
            "filters": filters,
            "atr_abs": atr_value,
            "signal_context": ctx_dict,
            "chosen_h": sig.get("chosen_h"),
            "chosen_t": sig.get("chosen_t"),
        }

        # ---- Trade guard ----
        trade_cfg = cfg.get("trade", {}) if "cfg" in locals() else {}
        trade_mode = (trade_cfg.get("mode") or "signal_only").lower()
        if trade_mode == "signal_only":
            result["_execution"] = "skipped(signal_only)"
            sig.setdefault("reason", "-")
            sig["_execution"] = "skipped(signal_only)"
        elif ctx.side in ("LONG", "SHORT"):
            try:
                last_px = float(df["close"].iloc[-1])
            except Exception:
                last_px = 0.0
            fixed_qty = float(trade_cfg.get("fixed_qty") or 0.0)
            notional = last_px * fixed_qty if (last_px and fixed_qty) else 0.0
            min_need = 0.0
            try:
                if get_min_notional is None:
                    raise RuntimeError("minnotional module not available")
                min_need = float(get_min_notional(symbol, cfg))
            except Exception:
                logger.warning(f"[WARN] cannot resolve min_notional for {symbol}, fallback=0")
                min_need = 0.0
            if notional < min_need:
                msg = f"min_notional_reject  need≥{min_need:.2f} USDT, got={notional:.2f}"
                result["_execution"] = "rejected(min_notional)"
                result["reason"] = f"{result['reason']}; {msg}" if result.get("reason") else msg
                filters.setdefault("extra_reasons", []).append("trade_guard=min_notional")
                sig["reason"] = msg
                sig["_execution"] = "rejected(min_notional)"
        # ---- /Trade guard ----

        ctx_dict["reason"] = result.get("reason")
        sig.update(
            {
                "side": ctx.side,
                "score": ctx.score,
                "threshold": float(threshold),
                "pt": ctx.pt,
                "sl": ctx.sl,
                "up_price": ctx.up_price,
                "down_price": ctx.down_price,
                "reason": result.get("reason"),
                "ts": bar_open_ts.isoformat(),
                "entry_price": ctx.entry_price,
                "signal_context": ctx_dict,
                "filters": filters,
                "prob_up_max": ctx.score,
                "prob_down_max": prob_down,
                "chosen_h": sig.get("chosen_h"),
                "chosen_t": sig.get("chosen_t"),
            }
        )
        if result.get("_execution"):
            sig["_execution"] = result["_execution"]

        return result
    except Exception as e:
        log_trace("LOOP_EXCEPTION", e)
        return {
            "side": "NONE",
            "score": 0.0,
            "reason": f"LOOP_EXCEPTION:{type(e).__name__}",
        }


def next_quarter_with_delay(now: datetime, delay_sec: int = 5) -> datetime:
    base = now.replace(second=0, microsecond=0)
    minute = (base.minute // 15) * 15
    slot = base.replace(minute=minute)
    if now >= slot + timedelta(seconds=delay_sec):
        slot += timedelta(minutes=15)
    return slot + timedelta(seconds=delay_sec)


def run_once(cfg: dict | str, delay_sec: int | None = None) -> dict:
    cfg = load_cfg(cfg)
    assert isinstance(cfg, dict), f"cfg must be dict, got {type(cfg)}"

    delay = 5 if delay_sec is None else int(delay_sec)
    if delay > 0:
        time.sleep(delay)
    now_utc_dt = datetime.now(timezone.utc)
    prev_open_dt, bar_open_dt = align_15m(now_utc=now_utc_dt, delay_sec=delay)
    prev_open_ts = pd.Timestamp(prev_open_dt)
    bar_open_ts = pd.Timestamp(bar_open_dt)
    logger.debug(
        "align_15m now=%s prev_open=%s bar_open=%s delay=%s",
        now_utc_dt.isoformat(),
        prev_open_dt.isoformat(),
        bar_open_dt.isoformat(),
        delay,
    )

    state = _load_dispatch_state()
    last_bar_open_raw = state.get("last_bar_open")
    last_bar_open_ts = None
    if last_bar_open_raw:
        try:
            last_bar_open_ts = pd.Timestamp(last_bar_open_raw)
        except Exception:
            logger.warning("[WARN] invalid last_bar_open in state: %s", last_bar_open_raw)

    if last_bar_open_ts is not None and last_bar_open_ts == bar_open_ts:
        logger.info(
            "skip dispatch: bar_open=%s already processed (state=%s)",
            bar_open_dt.isoformat(),
            STATE_FILE,
        )
        return {}

    host = socket.gethostname()
    try:
        if cfg.get("runtime", {}).get("notify", {}).get("telegram", False):
            tg_notify.notify(f"🟢 Realtime start (build={_BUILD}, host={host})")
        else:
            logger.info("notify: telegram disabled by cfg or env")
    except Exception as e:
        logger.warning("notify: startup swallow %s %s", type(e).__name__, e)
    telegram_conf = cfg.get("notify", {}).get("telegram")
    symbols = cfg.get("symbols", [])
    models = load_models(cfg)
    resources_dir = resolve_resources_dir(cfg)
    log_diag(
        f"realtime_loop: models_loaded={len(models)} resources_dir={resources_dir}"
    )
    thresholds_by_symbol = load_thresholds_by_symbol(cfg)
    default_threshold = float(cfg.get("train", {}).get("default_threshold", 0.6))
    horizon_bars = int(cfg.get("train", {}).get("target_horizon_bars", 16))
    now_utc_ts_pd = pd.Timestamp(now_utc_dt)
    results = {}
    os.makedirs("logs/diag", exist_ok=True)

    for sym in symbols:
        csv1 = os.path.join(resources_dir, f"{sym.split('USDT')[0].lower()}_15m.csv")
        csv2 = os.path.join(resources_dir, f"{sym.lower()}_15m.csv")
        csv_path = csv1 if os.path.exists(csv1) else csv2
        print(f"[REALTIME] {sym} <- {csv_path}")
        try:
            res = process_symbol(
                sym,
                cfg,
                models,
                csv_path,
                prev_open_ts=prev_open_ts,
                bar_open_ts=bar_open_ts,
                state=state,
                thresholds_by_symbol=thresholds_by_symbol,
                default_threshold=default_threshold,
                horizon_bars=horizon_bars,
                now_utc_ts=now_utc_ts_pd,
            )
        except Exception as e:
            log_trace("LOOP_EXCEPTION", e)
            res = {
                "side": "NONE",
                "score": 0.0,
                "reason": f"LOOP_EXCEPTION:{type(e).__name__}",
            }
        logger.debug(
            "dispatch_ctx symbol=%s now=%s prev_open=%s bar_open=%s prev_close=%s curr_open=%s src=%s",
            sym,
            now_utc_dt.isoformat(),
            prev_open_dt.isoformat(),
            bar_open_dt.isoformat(),
            res.get("prev_close"),
            res.get("price"),
            res.get("price_source", "-"),
        )
        sig = res if res.get("side") in ("LONG", "SHORT") else None
        if res.get("price") is not None and sig:
            notify_signal(sym, sig, float(res.get("price")), telegram_conf)
        # --- position sizing ---
        if res.get("side") in ("LONG", "SHORT") and res.get("_execution") != "skipped(signal_only)":
            ps_cfg = cfg.get("position_sizing", {})
            risk_cfg = cfg.get("risk", {})
            rule_cfg = ps_cfg.get("exchange_rule", {})
            rule = ExchangeRule(
                min_qty=float(rule_cfg.get("min_qty", 0)),
                qty_step=float(rule_cfg.get("qty_step", 0)),
                min_notional=float(rule_cfg.get("min_notional", 0)),
                max_leverage=int(rule_cfg.get("max_leverage", 1)),
            )
            equity = float(ps_cfg.get("equity_usdt", cfg.get("backtest", {}).get("initial_capital", 10000.0)))
            atr_abs = float(res.get("atr_abs", 0.0))
            tp_ratio = float(risk_cfg.get("take_profit_ratio", 0.0))
            sl_ratio = float(risk_cfg.get("stop_loss_ratio", 0.0))
            win_rate = ps_cfg.get("default_win_rate", None)
            inp = SizingInput(
                equity_usdt=equity,
                entry_price=float(res.get("price", 0.0)),
                atr_abs=atr_abs,
                side=res["side"],
                tp_ratio=tp_ratio,
                sl_ratio=sl_ratio,
                win_rate=win_rate,
                rule=rule,
            )
            qty = blended_sizing(
                inp,
                mode=ps_cfg.get("mode", "hybrid"),
                risk_per_trade=float(ps_cfg.get("risk_per_trade", 0.01)),
                atr_k=float(ps_cfg.get("atr_k", 1.5)),
                kelly_coef=float(ps_cfg.get("kelly_coef", 0.5)),
                kelly_floor=float(ps_cfg.get("kelly_floor", -0.5)),
                kelly_cap=float(ps_cfg.get("kelly_cap", 1.0)),
            )
            res["qty"] = qty
            res["sizing_mode"] = ps_cfg.get("mode", "hybrid")
            kelly_f = 0.0
            if win_rate is not None and sl_ratio > 0:
                kelly_f = kelly_fraction(win_rate, tp_ratio / sl_ratio)
            sizing_info = {
                "mode": ps_cfg.get("mode", "hybrid"),
                "equity_usdt": equity,
                "atr_abs": atr_abs,
                "risk_per_trade": float(ps_cfg.get("risk_per_trade", 0.01)),
                "tp_ratio": tp_ratio,
                "sl_ratio": sl_ratio,
                "kelly_f": kelly_f,
            }
            if qty > 0:
                notify_trade_open(
                    sym,
                    res["side"],
                    float(res.get("price", 0.0)),
                    qty,
                    sizing_info,
                    signal=res,
                    cfg=cfg,
                    tz="Asia/Taipei",
                )
            else:
                notify_guard(
                    "min_notional_reject",
                    {
                        "symbol": sym,
                        "side": res["side"],
                        "price": float(res.get("price", 0.0)),
                        "notional": float(res.get("price", 0.0)) * qty,
                        "min": rule.min_notional,
                    },
                    telegram_conf,
                )
        results[sym] = res
        if res.get("side") in ("LONG", "SHORT"):
            sig_state = _state_signals(state)
            sig_state[sym] = {
                "bar_open": bar_open_ts.isoformat(),
                "side": res.get("side"),
                "score": res.get("score"),
            }

    # snapshot if any bad scores
    bad = [
        r
        for r in results.values()
        if isinstance(r.get("score"), float)
        and (np.isnan(r["score"]) or not np.isfinite(r["score"]))
    ]
    if bad:
        snap = {"ts": datetime.utcnow().isoformat(), "bad": bad}
        with open("logs/diag/realtime_nan_snapshot.json", "w") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
        print("[DIAG] dumped logs/diag/realtime_nan_snapshot.json")

    def pct(p: Optional[float]) -> str:
        if p is None:
            return "-"
        try:
            return f"{float(p)*100:.2f}%"
        except Exception:
            return "-"

    def fmt_px(x: Optional[float]) -> str:
        if x is None:
            return "-"
        try:
            return f"{float(x):,.2f}"
        except Exception:
            return "-"

    multi: Dict[str, Dict[str, Any]] = {}
    for sym, d in results.items():
        ctx_dict = d.get("signal_context") or {}
        multi[sym] = {
            "symbol": sym,
            "side": ctx_dict.get("side", d.get("side")),
            "score": ctx_dict.get("score", sanitize_score(d.get("score"))),
            "threshold": ctx_dict.get("threshold", default_threshold),
            "h": ctx_dict.get("h_bars", d.get("horizon_bars")),
            "pt": ctx_dict.get("pt"),
            "sl": ctx_dict.get("sl"),
            "up_price": ctx_dict.get("up_price"),
            "down_price": ctx_dict.get("down_price"),
            "price": ctx_dict.get("entry_price", d.get("price")),
            "price_source": d.get("price_source"),
            "reason": d.get("reason"),
            "filters": d.get("filters"),
        }

    print(f"[NOTIFY] ⏱️ 多幣別即時訊號 (build={_BUILD}, host={host})")
    for sym in results:
        s = multi[sym]
        thr = s.get("threshold")
        try:
            thr_fmt = f"{float(thr):.2f}"
        except Exception:
            thr_fmt = "-"
        line = (
            f"{sym}: {s.get('side', 'NONE')} | "
            f"score={s.get('score', 0.0):.3f} (thr={thr_fmt}) | "
            f"h={s.get('h', '-')} | "
            f"pt={pct(s.get('pt'))} sl={pct(s.get('sl'))} | "
            f"↑={fmt_px(s.get('up_price'))} ↓={fmt_px(s.get('down_price'))} | "
            f"price={fmt_px(s.get('price'))}"
        )
        reason = s.get("reason")
        if reason:
            line += f" | reason={reason}"
        print(line)

    if cfg.get("runtime", {}).get("notify", {}).get("telegram", False):
        tg_notify.notify(multi, build=_BUILD, host=host)
    else:
        logger.info("notify: telegram disabled by cfg or env")

    print(json.dumps(multi, ensure_ascii=False, separators=(",", ": ")))
    state.update(
        {
            "last_bar_open": bar_open_dt.isoformat(),
            "last_prev_open": prev_open_dt.isoformat(),
            "last_dispatch_at": now_utc_dt.isoformat(),
        }
    )
    _save_dispatch_state(state)
    now_ts = datetime.now(tz=TW)
    for r in results.values():
        price = r.get("price")
        if price is not None:
            try:
                check_exit_once(cfg, float(price), now_ts)
            except Exception as e:
                print(f"[WARN] exit watchdog failed: {e}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Run realtime every 15m + delay seconds (with live Binance fetch).")
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--delay-sec", type=int, default=5)
    ap.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one cycle and exit (for systemd oneshot+timer).",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    if args.once:
        try:
            run_once(cfg, delay_sec=args.delay_sec)
        except Exception as e:
            log_trace("LOOP_EXCEPTION", e)
        sys.exit(0)

    while True:
        now = datetime.now(tz=TW)
        target = next_quarter_with_delay(now, args.delay_sec)
        wait = (target - now).total_seconds()
        if wait > 0:
            print(f"[LOOP] 現在 {now.strftime('%F %T%z')}，等到 {target.strftime('%F %T%z')} 再跑（{int(wait)} 秒）")
            time.sleep(wait)
        try:
            run_once(cfg, delay_sec=0)
        except Exception as e:
            log_trace("LOOP_EXCEPTION", e)


if __name__ == "__main__":
    main()
