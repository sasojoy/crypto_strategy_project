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

from csp.strategy import aggregator
from csp.strategy.model_hub import load_models
from csp.utils.paths import resolve_resources_dir
from csp.utils.timez import last_closed_15m, now_utc
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
# timezone utilities handled within aggregator
from csp.utils.validate_data import ensure_data_ready


def sanitize_score(x):
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return 0.0
        return xf
    except Exception:
        return 0.0

TW = tz.gettz("Asia/Taipei")
FRESH_MIN = 5.0  # 資料新鮮度門檻（分鐘）
logger = logging.getLogger(__name__)


def pick_latest_valid_row(features_df: pd.DataFrame, k: int = 3):
    """從最後 k 根中，挑選第一個「無缺值」的列；若都不合格，回傳 None。"""
    tail = features_df.tail(k)
    na_counts = tail.isna().sum()
    logging.info(f"[DIAG] tail_na_counts: {na_counts.to_dict()}")
    for idx in tail.index[::-1]:
        row = tail.loc[idx]
        if not row.isna().any() and np.isfinite(row.to_numpy(dtype=float)).all():
            return idx, row
    return None, None


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

    feature_cols = feat_params.get("feature_columns") or list(feats.columns)
    idx, row = pick_latest_valid_row(feats[feature_cols], k=3)
    if row is None:
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

def process_symbol(symbol: str, cfg: dict, models: dict):
    try:
        res = aggregator.read_or_fetch_latest(symbol, cfg=cfg)
        if isinstance(res, dict):
            return res
        df = res
        latest_close = df.index.max()
        anchor = last_closed_15m(now_utc())
        diff_min = (
            (anchor - latest_close).total_seconds() / 60.0
            if pd.notna(latest_close)
            else float("inf")
        )
        if pd.isna(latest_close) or diff_min >= FRESH_MIN:
            print(
                f"[STALE] {symbol}: anchor={anchor.isoformat()} latest_close={latest_close.isoformat() if pd.notna(latest_close) else 'none'} diff_min={diff_min:.2f}"
            )
            return {"side": "NONE", "score": 0.0, "reason": "stale_data"}
        bundle = models.get(symbol)
        if not bundle:
            return {"side": "NONE", "score": 0.0, "reason": "no_models_loaded"}
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
    rt_cfg = cfg.get("realtime", {})
    fetch_mode = str(rt_cfg.get("fetch_mode", "")).lower()
    fetch_disabled = fetch_mode in ("", "none", "csv_only", "csv_only_or_none")
    resources_dir = resolve_resources_dir(cfg)
    log_diag(
        f"realtime_loop: models_loaded={len(models)} resources_dir={resources_dir}"
    )
    results = {}
    os.makedirs("logs/diag", exist_ok=True)

    for sym in symbols:
        csv_path = aggregator._csv_path_for_symbol(sym, cfg)
        if not csv_path:
            print(f"[SKIP] {sym}: csv not found under resources_dir")
            continue
        print(f"[REALTIME] {sym} <- {csv_path}")
        if not fetch_disabled:
            try:
                ensure_data_ready(sym, csv_path)
            except Exception as fe:
                print(f"[WARN] {sym}: data fetch failed: {fe}")
        try:
            res = process_symbol(sym, cfg, models)
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
