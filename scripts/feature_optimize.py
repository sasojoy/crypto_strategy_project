from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from csp.data.loader import load_15m_csv
from csp.utils.dates import resolve_time_range_like
from csp.optimize.feature_opt import optimize_symbol


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
    ap.add_argument("--symbols", nargs="*", default=None, help="指定要跑的幣別")
    ap.add_argument("--days", type=int, default=None, help="回測天數（未指定 start/end 時生效）")
    ap.add_argument("--trials", type=int, default=20, help="Optuna trials 數")
    ap.add_argument("--outdir", default="reports/feature_opt", help="輸出目錄")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))
    symbols = args.symbols or cfg.get("symbols", [])
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
        idx = pd.DatetimeIndex(df["timestamp"])
        start_ts, end_ts = resolve_time_range_like(env_args, idx)
        print(f"[OPT] {sym} {start_ts}~{end_ts} trials={args.trials}")
        study = optimize_symbol(args.cfg, sym, start_ts, end_ts, args.trials, base_out)
        print(f"[BEST {sym}] value={study.best_value:.6f} params={study.best_params}")


if __name__ == "__main__":
    main()
