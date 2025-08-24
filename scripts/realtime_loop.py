from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta

import math
import pandas as pd
from dateutil import tz

from csp.data.fetcher import update_csv_with_latest
from csp.strategy.aggregator import get_latest_signal
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

TW = tz.gettz("Asia/Taipei")

FRESH_MIN = 5.0  # 資料新鮮度門檻（分鐘）


def process_symbol(symbol: str, cfg: dict):
    # 1) 檢查資料新鮮度
    csv_path = cfg["io"]["csv_paths"][symbol]
    df = pd.read_csv(csv_path)
    ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
    age_min = (pd.Timestamp.utcnow() - ts).total_seconds() / 60.0
    if age_min > FRESH_MIN:
        notify_guard(
            "data_lag",
            {"symbol": symbol, "age_min": round(age_min, 2), "last_ts": str(ts)},
        )
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "data_lag"}

    # 2) 取得最新訊號（內部會嚴格比對 meta.feature_columns 與特徵 NaN）
    sig = get_latest_signal(symbol, cfg, fresh_min=FRESH_MIN)
    if not sig:
        notify_guard("signal_unavailable", {"symbol": symbol})
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "signal_unavailable"}

    # 3) 格式化得分，避免 NaN
    score = 0.0
    s = sig.get("score")
    if s is not None and not (isinstance(s, float) and math.isnan(s)):
        score = float(s)
    sig["score"] = score
    return sig


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
    csv_map = cfg.get("io", {}).get("csv_paths", {})
    live_cfg = (cfg.get("io", {}) or {}).get("live_fetch", {}) or {}
    results = {}

    for sym in symbols:
        csv_path = csv_map.get(sym)
        if not csv_path:
            print(f"[SKIP] {sym}: No CSV path in config")
            continue
        print(f"[REALTIME] {sym} <- {csv_path}")
        stale = False
        df = None
        if live_cfg.get("enabled"):
            try:
                df = update_csv_with_latest(sym, csv_path, interval=live_cfg.get("interval", "15m"))
                last_ts = df["timestamp"].iloc[-1]
                print(f"  last closed UTC={last_ts.isoformat()} | TW={(last_ts.tz_convert(TW)).isoformat()}")
                stale = bool(df.attrs.get("stale"))
            except Exception as e:
                print(f"[WARN] live fetch failed for {sym}: {e}")
                stale = True
        if df is None:
            try:
                df = pd.read_csv(csv_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            except Exception as e:
                print(f"[WARN] load csv failed for {sym}: {e}")
                df = None
        last_price = None
        if df is not None:
            try:
                last_price = float(df["close"].iloc[-1])
            except Exception:
                pass
        try:
            res = process_symbol(sym, cfg)
        except Exception as e:
            res = {"symbol": sym, "side": "NONE", "error": str(e)}
        sig = res if res.get("side") in ("LONG", "SHORT") else None
        if last_price is not None:
            res["price"] = last_price
        if sig and last_price is not None:
            notify_signal(sym, sig, last_price, telegram_conf)
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
        if stale:
            res["warning"] = "STALE DATA"
            notify_guard("data_lag", {"symbol": sym}, telegram_conf)
        results[sym] = res

    lines = []
    for sym, r in results.items():
        if "error" in r:
            lines.append(f"{sym}: ERROR {r['error']}")
        else:
            side = r.get("side") or "-"
            score = r.get("score", float("nan"))
            reason = r.get("reason", "-")
            note = " [STALE DATA]" if r.get("warning") else ""
            lines.append(f"{sym}: {side} | score={score:.3f} | reason={reason}{note}")
    notify("⏱️ 多幣別即時訊號\n" + "\n".join(lines), cfg.get("notify", {}).get("telegram"))

    print(json.dumps(results, ensure_ascii=False, indent=2))
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
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
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
            print(f"[ERROR] loop run failed: {e}")


if __name__ == "__main__":
    main()
