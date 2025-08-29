from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone

from dateutil import tz

from csp.strategy.aggregator import get_latest_signal, sanitize_score
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
from csp.utils.tz import ensure_utc_index, ensure_utc_ts, now_utc as _now_utc, floor_to

TW = tz.gettz("Asia/Taipei")

FRESH_MIN = 5.0  # 資料新鮮度門檻（分鐘）

# --- ADD: live fetch helper (no extra deps needed) ---
import math
import pandas as pd
import urllib.request, urllib.parse, json as _json

BINANCE_BASE = "https://api.binance.com"
INTERVAL = "15m"

def _binance_klines(symbol: str, interval: str, end_ms: int | None, limit: int = 720):
    """
    Public klines; endTime inclusive-ish per Binance semantics.
    """
    qs = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_ms:
        qs["endTime"] = end_ms
    url = f"{BINANCE_BASE}/api/v3/klines?{urllib.parse.urlencode(qs)}"
    with urllib.request.urlopen(url, timeout=15) as r:
        return _json.loads(r.read().decode("utf-8"))

def ensure_latest_csv(symbol: str, csv_path: str, fresh_min: float = 5.0):
    """
    將 csv 補到「現在 floor(15m)」。
    """
    try:
        df_old = pd.read_csv(csv_path)
    except FileNotFoundError:
        df_old = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    df_old = ensure_utc_index(df_old, ts_col="timestamp")
    print(f"[DIAG] df.index.tz={df_old.index.tz}, head_ts={df_old.index[:3].tolist()}")
    assert str(df_old.index.tz) == "UTC", "[DIAG] index not UTC"

    now_ts = _now_utc()
    floor_now = floor_to(now_ts, "15min")

    last_ts = df_old.index.max() if len(df_old) else pd.NaT

    if pd.notna(last_ts):
        lag_min = (now_ts - last_ts).total_seconds() / 60.0
        if lag_min <= (15.0 + fresh_min):
            return False

    # 以 floor_now 當作 endTime（ms）
    end_ms = int(floor_now.value // 10**6)
    # 多抓一點避免斷帶（~30 天 ≈ 2880 根）
    kl = _binance_klines(symbol, INTERVAL, end_ms=end_ms, limit=2880)
    if not kl:
        return False

    # 轉成 DataFrame
    tmp = pd.DataFrame(kl, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "_q","_n","_taker","_taker_vol","_i"
    ])
    # Binance 的 open_time/close_time 是 ms；「理論收盤點」= open_time + 15m
    tmp["open_time"] = pd.to_datetime(tmp["open_time"], unit="ms", utc=True)
    tmp["timestamp"] = tmp["open_time"] + pd.Timedelta(minutes=15)
    for col in ("open","high","low","close","volume"):
        tmp[col] = tmp[col].astype(float)

    tmp = tmp[["timestamp","open","high","low","close","volume"]]
    tmp = ensure_utc_index(tmp, ts_col="timestamp")

    merged = pd.concat([df_old, tmp])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    merged.reset_index().to_csv(csv_path, index=False)
    return True

def process_symbol(symbol: str, cfg: dict):
    sig = get_latest_signal(symbol, cfg, fresh_min=FRESH_MIN)
    if not sig:
        notify_guard("signal_unavailable", {"symbol": symbol})
        return {"symbol": symbol, "side": "NONE", "score": 0.0, "reason": "signal_unavailable"}
    sig["score"] = sanitize_score(sig.get("score"))
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
    results = {}

    for sym in symbols:
        csv_path = csv_map.get(sym)
        if not csv_path:
            print(f"[SKIP] {sym}: No CSV path in config")
            continue
        print(f"[REALTIME] {sym} <- {csv_path}")
        # --- ADD: 先把 CSV 補到最新，再算訊號 ---
        try:
            updated = ensure_latest_csv(sym, csv_path, fresh_min=FRESH_MIN)
            if updated:
                print(f"[FETCH] {sym}: csv updated to latest 15m close.")
        except Exception as fe:
            print(f"[WARN] {sym}: live fetch failed: {fe}")
        try:
            res = process_symbol(sym, cfg)
        except Exception as e:
            res = {"symbol": sym, "side": "NONE", "error": str(e)}
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
