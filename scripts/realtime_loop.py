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

from typing import List, Tuple, Dict, Any

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


def ensure_not_stale(df: "pd.DataFrame", symbol: str, now_utc: "pd.Timestamp") -> Tuple[bool, str]:
    """檢查 df 是否有覆蓋到最新已收盤 anchor，一旦落後就回傳 (False, reason)。"""
    if df.empty:
        return False, "empty_df"
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
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

TW = tz.gettz("Asia/Taipei")
logger = logging.getLogger(__name__)



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
    return ", ".join([f"h{str(k)}={v:.2f}" for k, v in items])


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

    side = "LONG" if score >= 0.75 else ("SHORT" if score <= 0.25 else "NONE")
    out = {"symbol": symbol, "side": side, "score": score}
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
            allow_one = cfg.get("realtime", {}).get("allow_stale_one_bar", True)
            if allow_one and "diff_min=15.00" in reason:
                logger.info(
                    f"[WARN] {symbol} {reason} — allow_stale_one_bar=True, proceed with last_close"
                )
            else:
                return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": reason}

        bundle = models.get(symbol)
        if not bundle:
            return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "no_models_loaded"}
        model = bundle.get("model") if isinstance(bundle, dict) else bundle
        scaler = bundle.get("scaler") if isinstance(bundle, dict) else None
        sig = predict_one(symbol, df, model, scaler, cfg_path="csp/configs/strategy.yaml")
        # ---- Trade guard ----
        trade_cfg = cfg.get("trade", {}) if 'cfg' in locals() else {}
        trade_mode = (trade_cfg.get("mode") or "signal_only").lower()
        if trade_mode == "signal_only":
            sig.setdefault("reason", "-")
            sig["_execution"] = "skipped(signal_only)"
        else:
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
                sig["reason"] = f"min_notional_reject  need≥{min_need:.2f} USDT, got={notional:.2f}"
                sig["_execution"] = "rejected(min_notional)"
        # ---- /Trade guard ----
        side = sig.get("side", "NONE")
        score = sanitize_score(sig.get("score"))
        result = {
            "symbol": symbol,
            "side": side,
            "score": score,
            "price": float(df["close"].iloc[-1]) if not df.empty else None,
        }
        if side == "NONE":
            result["reason"] = sig.get("reason", "below_threshold")
        if sig.get("_execution"):
            result["_execution"] = sig["_execution"]
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

    formatted_lines = [format_signal_summary(results[sym]) for sym in results]
    logger.info("[NOTIFY] ⏱️ 多幣別即時訊號")
    for line in formatted_lines:
        print(line)
    notify("⏱️ 多幣別即時訊號\n" + "\n".join(formatted_lines), cfg.get("notify", {}).get("telegram"))

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
