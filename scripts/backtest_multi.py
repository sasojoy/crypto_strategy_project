# scripts/backtest_multi.py
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# 依賴 backtest_v2 的核心邏輯
from csp.backtesting.backtest_v2 import run_backtest_for_symbol
from csp.metrics.report import summarize
from csp.utils.io import load_cfg
from csp.utils.tz_safe import (
    normalize_df_to_utc,
    safe_ts_to_utc,
    now_utc,
    floor_utc,
)

BINANCE_BASE = "https://api.binance.com"

def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def read_local_csv(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    df = pd.read_csv(p)
    df = normalize_df_to_utc(df)
    if "timestamp" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 timestamp 欄位")
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"{csv_path} 缺少必要欄位：{c}")
    df = normalize_df_to_utc(df)
    print(f"[DIAG] df.index.tz={df.index.tz}, head_ts={df.index[:3].tolist()}")
    assert str(df.index.tz) == "UTC", "[DIAG] index not UTC"
    return df

def write_local_csv(csv_path: str, df: pd.DataFrame) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

def fetch_binance_klines(symbol: str, interval: str, start_ts_ms: int, end_ts_ms: Optional[int]=None, limit: int=1000) -> List[List]:
    """
    呼叫 Binance Klines API，回傳 K 線陣列（每筆：[open_time, open, high, low, close, volume, close_time, ...]）
    參考：https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
    """
    params = {"symbol": symbol, "interval": interval, "startTime": start_ts_ms, "limit": limit}
    if end_ts_ms is not None:
        params["endTime"] = end_ts_ms
    r = requests.get(f"{BINANCE_BASE}/api/v3/klines", params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def append_missing_15m(csv_path: str, symbol: str, end_utc: datetime) -> None:
    """
    若本地 CSV 缺資料，從最後一根後補到 end_utc（含）
    """
    df = read_local_csv(csv_path)
    interval_ms = 15 * 60 * 1000
    if df.empty:
        # 沒有就從 end_utc 往回抓 360 天（保守值，可調成 args.days）
        end_ms = int(end_utc.timestamp() * 1000)
        start_ms = end_ms - 360 * 24 * 60 * 60 * 1000
    else:
        last_ts_ms = int(df["timestamp"].iloc[-1].timestamp() * 1000)
        start_ms = last_ts_ms + interval_ms
        end_ms = int(end_utc.timestamp() * 1000)
        if start_ms > end_ms:
            return  # 已最新

    rows: List[Dict] = []
    cursor = start_ms
    while cursor <= end_ms:
        batch = fetch_binance_klines(symbol, "15m", cursor, end_ms)
        if not batch:
            break
        for k in batch:
            open_time_ms = int(k[0])
            if open_time_ms > end_ms:
                break
            rows.append({
                "timestamp": pd.to_datetime(open_time_ms, unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        # 前進
        cursor = int(batch[-1][0]) + 15*60*1000
        if len(batch) < 1000:
            break

    if rows:
        df_new = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        df_old = read_local_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        write_local_csv(csv_path, df_all)

def slice_by_days(csv_path: str, days: int, end_utc: datetime) -> Tuple[pd.Timestamp, pd.Timestamp]:
    end_ts = pd.Timestamp(end_utc)
    start_ts = end_ts - pd.Timedelta(days=days)
    return (safe_ts_to_utc(start_ts), safe_ts_to_utc(end_ts))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="path to strategy.yaml")
    ap.add_argument("--days", type=int, default=30, help="回測天數（會切資料區間）")
    ap.add_argument("--fetch", choices=["auto","none"], default="auto", help="是否自動補資料（Binance）")
    ap.add_argument("--save-summary", action="store_true", help="啟用報表輸出")
    ap.add_argument("--out-dir", default="reports", help="輸出目錄（預設 reports）")
    ap.add_argument("--format", choices=["csv", "json", "both"], default="both", help="輸出格式（csv, json, both）")
    ap.add_argument("--export-equity-bars", action="store_true", help="輸出每根K線的資金曲線CSV（equity_curve_bars_*.csv）")
    ap.add_argument("--plot-equity", action="store_true", help="輸出每根K線資金曲線的 PNG 圖")
    ap.add_argument("--symbols", nargs="*", default=None, help="指定要跑的幣別（預設讀 cfg.symbols）")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    symbols: List[str] = args.symbols or cfg.get("symbols", [])
    if not symbols:
        print("cfg.symbols 為空，請在 strategy.yaml 設定幣別")
        sys.exit(1)

    base_out_dir = Path(args.out_dir)
    run_id = now_utc().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out_dir / run_id if args.save_summary else base_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    end_utc_dt = now_utc()

    # 1) 自動補資料（如開啟）
    if args.fetch == "auto":
        for sym in symbols:
            csv_path = cfg["io"]["csv_paths"][sym]
            print(f"[FETCH] {sym} -> {csv_path}")
            try:
                append_missing_15m(csv_path, sym, end_utc_dt)
            except Exception as e:
                print(f"[WARN] fetch {sym} 失敗：{e}（略過，使用現有 CSV）")

    # 2) 依 days 產生時間窗
    start_ts, end_ts = slice_by_days("dummy.csv", args.days, end_utc_dt)
    print(f"[WINDOW] start={start_ts}  end={end_ts}  days={args.days}")

    # 3) 執行回測
    summary_all: Dict[str, dict] = {}
    for sym in symbols:
        csv_path = cfg["io"]["csv_paths"][sym]
        print(f"[RUN] {sym} days={args.days} csv={csv_path}")
        res = run_backtest_for_symbol(csv_path, args.cfg, symbol=sym, start_ts=start_ts, end_ts=end_ts)

        trades_df = res.get("trades", pd.DataFrame())
        eq_df = res.get("equity_curve", pd.DataFrame())

        # 3.1（可選）輸出每根K線的資金曲線
        if args.plot_equity and isinstance(eq_df, pd.DataFrame) and not eq_df.empty:
            try:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                plt.plot(eq_df["timestamp"], eq_df.get("equity", eq_df.get("equity_after")))
                plt.title(f"Equity Curve (Bars) - {sym}")
                plt.xlabel("Time")
                plt.ylabel("Equity")
                png_path = out_dir / f"equity_curve_bars_{sym}.png"
                fig.savefig(png_path, dpi=144, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] plot {sym} 失敗：{e}")

        if args.export_equity_bars and isinstance(eq_df, pd.DataFrame) and not eq_df.empty:
            eq_path = out_dir / f"equity_curve_bars_{sym}.csv"
            eq_df.to_csv(eq_path, index=False, encoding="utf-8-sig")

        # 3.2 計算 summary
        metrics = summarize(eq_df, trades_df, bar_seconds=900)
        summary_all[sym] = metrics
        print(f"[SUMMARY {sym}] " + json.dumps(metrics, ensure_ascii=False))

        # 3.3 輸出 trades 及 summary
        if args.save_summary:
            trades_path = out_dir / f"trades_{sym}.csv"
            trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
            if args.format in ("json", "both"):
                (out_dir / f"summary_{sym}.json").write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
            if args.format in ("csv", "both"):
                pd.DataFrame([metrics]).to_csv(
                    out_dir / f"summary_{sym}.csv", index=False, encoding="utf-8-sig"
                )

    # 4) 合併輸出
    if args.save_summary:
        (out_dir / "summary_all.json").write_text(
            json.dumps(summary_all, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    print("[SUMMARY ALL] " + json.dumps(summary_all, ensure_ascii=False))
    print(f"[DONE] outputs -> {out_dir}")

if __name__ == "__main__":
    main()
