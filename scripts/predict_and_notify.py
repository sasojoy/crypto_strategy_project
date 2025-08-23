from __future__ import annotations
import argparse
import json

from csp.pipeline.realtime_v2 import run_once
from csp.utils.notifier import notify
from csp.utils.io import load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to 15m CSV (latest rows used)")
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    res = run_once(args.csv, cfg, debug=args.debug)

    if "error" in res:
        line = f"{res.get('symbol', '?')}: ERROR {res['error']}"
    else:
        side_display = res["side"].upper() if res.get("side") else "WAIT"
        line = f"{res['symbol']}: {side_display} | P={res['price']:.2f} | proba_up={res['proba_up']:.3f}"
        if res.get("side"):
            line += f" | TP={res['tp']:.2f} | SL={res['sl']:.2f}"
        if res.get("diag_low_var"):
            line += " [DIAG:LOW VAR]"
    notify(line, cfg.get("notify", {}).get("telegram"))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
