from __future__ import annotations

import argparse
import json
import time
import os
import sys
import math
import traceback

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dateutil import tz

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

from csp.strategy.model_hub import load_models
from csp.utils.paths import resolve_resources_dir
from csp.utils.timez import now_utc, UTC
from csp.strategy.position_sizing import (
    blended_sizing, SizingInput, ExchangeRule, kelly_fraction
)
from csp.utils.notifier import (
    notify,
    notify_signal,
    notify_trade_open,
    notify_guard,
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


def ensure_not_stale(df: "pd.DataFrame", symbol: str, now_utc: "pd.Timestamp") -> tuple[bool, str]:
    """檢查 df 是否有覆蓋到最新已收盤 anchor，一旦落後就回傳 (False, reason)。"""
    if df.empty:
        return False, "empty_df"
    # 假設 df 的索引是 tz-aware UTC 的 datetimeindex；若不是要先 to_datetime + tz_localize('UTC')
    last_close = df.index.max()
    anchor = ensure_latest_closed_anchor(now_utc)
    diff_min = (anchor - last_close).total_seconds() / 60.0
    if diff_min > 0:
        return False, f"stale_data(anchor={anchor.isoformat()}, latest_close={last_close.isoformat()}, diff_min={diff_min:.2f})"
    return True, ""


def validate_numeric_features(feats: "pd.DataFrame", feature_cols: list[str]) -> None:
    bad = []
    for c in feature_cols:
        if c not in feats.columns:
            bad.append(f"{c} (missing)")
        else:
            if not pd.api.types.is_numeric_dtype(feats[c].dtype):
                bad.append(f"{c} ({feats[c].dtype})")
    if bad:
        raise ValueError("Non-numeric or missing feature columns detected: " + ", ".join(bad))


def sanitize_score(x):
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return 0.0
        return xf
    except Exception:
        return 0.0

TW = tz.gettz("Asia/Taipei")
logger = logging.getLogger(__name__)


def pick_latest_valid_row(df_feat: "pd.DataFrame", k: int = 3):
    # 只選 numeric 欄位做有限值判斷
    numeric_df = df_feat.select_dtypes(include=[np.number])
    for idx in range(len(df_feat) - 1, -1, -1):
        row_full = df_feat.iloc[idx]
        row_num = numeric_df.iloc[idx]
        if not row_num.isna().any() and np.isfinite(row_num.to_numpy(dtype=float)).all():
            return idx, row_full
    raise RuntimeError("No valid feature row found in last k bars")


def predict_one(symbol: str, df_15m: pd.DataFrame, model, scaler, cfg_path: str = "csp/configs/strategy.yaml"):
    """Build features, pick latest valid row and run prediction."""
    from csp.features.h16 import build_features_15m_4h
    from csp.core.feature import add_features
    from csp.utils.config import get_symbol_features

    cfg = load_cfg(cfg_path)
    feat_params = get_symbol_features(cfg, symbol)

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
    feature_cols = feat_params.get("feature_columns") or list(feats.columns)
    # 僅用於特徵的 DataFrame（避免 timestamp 等混進來）
    X = feats[feature_cols].copy()
    validate_numeric_features(X, feature_cols)
    try:
        idx, row = pick_latest_valid_row(X, k=3)
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

    x = row.to_frame().T
    try:
        x2 = scaler.transform(x) if scaler is not None else x.values
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

    side = "LONG" if score >= 0.75 else ("SHORT" if score <= 0.25 else "NONE")
    return {"symbol": symbol, "side": side, "score": score}


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

def process_symbol(symbol: str, cfg: dict, models: dict, csv_path: str):
    try:
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"], utc=True)
            df = df.drop(columns=["timestamp"])
        else:
            df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        ok, reason = ensure_not_stale(df, symbol, pd.Timestamp.now(tz=UTC))
        if not ok:
            fetch_cfg = cfg.get("fetch", {}) or {}
            fetch_mode = fetch_cfg.get("mode", "csv_only")
            if fetch_mode not in ("none", "csv_only"):
                df, _ = read_or_fetch_latest(cfg, symbol, csv_path, now_ts_in=None)
                df.index = pd.to_datetime(df.index, utc=True)
                df = df.sort_index()
                ok, reason = ensure_not_stale(df, symbol, pd.Timestamp.now(tz=UTC))
                if not ok:
                    return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": reason}
            else:
                return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": reason}

        bundle = models.get(symbol)
        if not bundle:
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}
        model = bundle.get("model") if isinstance(bundle, dict) else bundle
        scaler = bundle.get("scaler") if isinstance(bundle, dict) else None
        sig = predict_one(symbol, df, model, scaler, cfg_path="csp/configs/strategy.yaml")
        side = sig.get("side", "NONE")
        score = sanitize_score(sig.get("score"))
        result = {
            "side": side,
            "score": score,
            "price": float(df["close"].iloc[-1]) if not df.empty else None,
        }
        if side == "NONE":
            result["reason"] = "below_threshold"
        return result
    except Exception as e:
        log_trace("LOOP_EXCEPTION", e)
        return {
            "side": "NONE",
            "score": 0.0,
            "reason": f"LOOP_EXCEPTION:{type(e).__name__}",
        }


def next_quarter_with_delay(now: datetime, delay_sec: int = 15) -> datetime:
    base = now.replace(second=0, microsecond=0)
    minute = (base.minute // 15) * 15
    slot = base.replace(minute=minute)
    if now >= slot + timedelta(seconds=delay_sec):
        slot += timedelta(minutes=15)
    return slot + timedelta(seconds=delay_sec)


def run_once(cfg: dict | str, delay_sec: int | None = None) -> dict:
    cfg = load_cfg(cfg)
    assert isinstance(cfg, dict), f"cfg must be dict, got {type(cfg)}"
    telegram_conf = cfg.get("notify", {}).get("telegram")
    symbols = cfg.get("symbols", [])
    models = load_models(cfg)
    resources_dir = resolve_resources_dir(cfg)
    log_diag(
        f"realtime_loop: models_loaded={len(models)} resources_dir={resources_dir}"
    )
    results = {}
    os.makedirs("logs/diag", exist_ok=True)

    for sym in symbols:
        csv1 = os.path.join(resources_dir, f"{sym.split('USDT')[0].lower()}_15m.csv")
        csv2 = os.path.join(resources_dir, f"{sym.lower()}_15m.csv")
        csv_path = csv1 if os.path.exists(csv1) else csv2
        print(f"[REALTIME] {sym} <- {csv_path}")
        try:
            res = process_symbol(sym, cfg, models, csv_path)
        except Exception as e:
            log_trace("LOOP_EXCEPTION", e)
            res = {
                "side": "NONE",
                "score": 0.0,
                "reason": f"LOOP_EXCEPTION:{type(e).__name__}",
            }
        sig = res if res.get("side") in ("LONG", "SHORT") else None
        if res.get("price") is not None and sig:
            notify_signal(sym, sig, float(res.get("price")), telegram_conf)
        # --- position sizing ---
        if res.get("side") in ("LONG", "SHORT"):
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

    lines = []
    for sym, r in results.items():
        if "error" in r:
            lines.append(f"{sym}: ERROR {r['error']}")
        else:
            side = r.get("side") or "NONE"
            scr = sanitize_score(r.get("score"))
            reason = r.get("reason", "-")
            note = " [STALE DATA]" if r.get("warning") else ""
            # 嚴禁 NaN 外流
            try:
                scr = float(scr)
                if math.isnan(scr):
                    scr = 0.0
            except Exception:
                scr = 0.0
            lines.append(f"{sym}: {side} | score={scr:.3f} | reason={reason}{note}")
    notify("⏱️ 多幣別即時訊號\n" + "\n".join(lines), cfg.get("notify", {}).get("telegram"))

    payload = {
        sym: {
            "symbol": sym,
            "side": d.get("side", "NONE"),
            "score": sanitize_score(d.get("score")),
            "reason": d.get("reason"),
        }
        for sym, d in results.items()
    }
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ": ")))
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
    ap.add_argument("--delay-sec", type=int, default=15)
    ap.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one cycle and exit (for systemd oneshot+timer).",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    if args.once:
        try:
            run_once(cfg)
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
            run_once(cfg)
        except Exception as e:
            log_trace("LOOP_EXCEPTION", e)


if __name__ == "__main__":
    main()
