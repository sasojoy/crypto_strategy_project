from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from csp.data.loader import load_15m_csv
from csp.utils.dates import resolve_time_range_like
from csp.optimize.feature_opt import optimize_symbol, apply_best_params_to_cfg
from csp.utils.io import load_cfg


def _read_date_args_from_env() -> dict:
    start = os.getenv("START_DATE")
    end = os.getenv("END_DATE")
    days = os.getenv("DAYS")
    if days is not None:
        try:
            days = int(days)
        except Exception:
            days = None
    return {"start": start, "end": end, "days": days}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="path to strategy.yaml")
    ap.add_argument("--symbols", default=None, help="指定要跑的幣別 (逗號分隔)")
    ap.add_argument("--days", type=int, default=None, help="回測天數（未指定 start/end 時生效）")
    ap.add_argument("--trials", type=int, default=20, help="Optuna trials 數")
    ap.add_argument("--outdir", default="reports/feature_opt", help="輸出目錄")
    ap.add_argument("--apply-to-cfg", action="store_true", help="將最佳參數寫回 cfg")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    else:
        symbols = cfg.get("symbols", [])
    if not symbols:
        print("cfg.symbols 為空，請在 strategy.yaml 設定幣別")
        sys.exit(1)

    base_out = Path(args.outdir)
    base_out.mkdir(parents=True, exist_ok=True)

    env_args = _read_date_args_from_env()
    if args.days is not None:
        env_args["days"] = args.days

    for sym in symbols:
        csv_path = cfg["io"]["csv_paths"][sym]
        df = load_15m_csv(csv_path)
        # Ensure a timestamp column exists for legacy paths
        if "timestamp" not in df.columns:
            df["timestamp"] = df.index

        # Robust timestamp extraction: column or index

        def _get_utc_index(df):
            # Case A: explicit 'timestamp' column exists
            if "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"], utc=True)
                return pd.DatetimeIndex(ts, tz="UTC")
            # Case B: already on DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is None:
                    return df.index.tz_localize("UTC")
                return df.index.tz_convert("UTC")
            # Case C: fallback - try to coerce an index to datetime
            idx = pd.to_datetime(df.index, utc=True, errors="raise")
            return pd.DatetimeIndex(idx, tz="UTC")

        idx = _get_utc_index(df)
        # DIAG
        print(f"[DIAG] feature_optimize: idx.tz={getattr(idx, 'tz', None)}, has_col={'timestamp' in df.columns}, columns={list(df.columns)[:8]}...")

        df = df.set_index(idx)
        assert str(df.index.tz) == "UTC", "[DIAG] feature_optimize: index must be UTC"

        start_ts, end_ts = resolve_time_range_like(env_args, idx)
        print(f"[OPT] {sym} {start_ts}~{end_ts} trials={args.trials}")
        study = optimize_symbol(args.cfg, sym, start_ts, end_ts, args.trials, base_out)
        print(f"[BEST {sym}] value={study.best_value:.6f} params={study.best_params}")
        apply_best_params_to_cfg(args.cfg, sym, study.best_params,
                                 apply=args.apply_to_cfg,
                                 log_file=base_out / "apply_log.txt")


if __name__ == "__main__":
    main()
