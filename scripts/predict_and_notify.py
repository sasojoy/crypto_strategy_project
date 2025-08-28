from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd

from csp.pipeline.realtime_v2 import run_once
from csp.utils.notifier import notify as base_notify
from csp.utils.io import load_cfg


def notify(message, telegram_cfg, *, score=None, x_last=None):
    if score is not None and np.isnan(score):
        series = pd.Series(x_last)
        nan_cols = series[series.isna()].index.tolist()
        print(f"[DIAG] Signal=NONE | score=nan | reason=NAN_FEATURES | cols={nan_cols}")
    base_notify(message, telegram_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Path to 15m CSV (latest rows used)")
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    csv_path = args.csv
    if not csv_path:
        csv_paths = cfg.get("io", {}).get("csv_paths", {})
        csv_path = next(iter(csv_paths.values()), None)
        if not csv_path:
            raise ValueError("--csv not provided and cfg.io.csv_paths empty")
    res = run_once(csv_path, cfg, debug=args.debug)

    if "error" in res:
        line = f"{res.get('symbol', '?')}: ERROR {res['error']}"
    else:
        side_display = res["side"].upper() if res.get("side") else "WAIT"
        line = f"{res['symbol']}: {side_display} | P={res['price']:.2f} | proba_up={res['proba_up']:.3f}"
        if res.get("side"):
            line += f" | TP={res['tp']:.2f} | SL={res['sl']:.2f}"
        if res.get("diag_low_var"):
            line += " [DIAG:LOW VAR]"
    notify(
        line,
        cfg.get("notify", {}).get("telegram"),
        score=res.get("score"),
        x_last=res.get("diag_X_last"),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
