# scripts/run_bt_v2.py
import argparse, json
from pathlib import Path
from csp.backtesting.backtest_v2 import run_backtest_for_symbol
from csp.utils.io import load_cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="csp/configs/strategy.yaml")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--symbol", default="ALL")  # BTCUSDT/ETHUSDT/BCHUSDT/ALL
    args = ap.parse_args()

    cfg_path = args.cfg
    cfg = load_cfg(cfg_path)
    symbols = cfg.get("symbols", [])
    targets = symbols if args.symbol.upper() == "ALL" else [args.symbol]

    out_dir = Path("backtests"); out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for sym in targets:
        csv_path = cfg["io"]["csv_paths"][sym]
        print(f"[RUN] {sym} days={args.days} csv={csv_path}")
        r = run_backtest_for_symbol(csv_path, cfg, symbol=sym)
        results[sym] = r["metrics"]
        # 存交易明細
        (out_dir / f"{sym}_trades.csv").write_text(
            r["trades"].to_csv(index=False, encoding="utf-8-sig"),
            encoding="utf-8"
        )
        # 存單幣別摘要
        (out_dir / f"{sym}_summary.json").write_text(
            json.dumps(r["metrics"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] {sym} -> backtests\\{sym}_trades.csv")

    # 存合併摘要
    (out_dir / "summary_all.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] summary -> backtests\\summary_all.json")

if __name__ == "__main__":
    main()
