import sys, os, json, joblib, pickle, time
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.append(os.getcwd())
from strategy.logic import TradingStrategy

def send_telegram(msg):
    print(f'[TELEGRAM] {msg}')

def run_production_cycle():
    strategy = TradingStrategy()
    print(f'--- H16 DUAL_TRACK_LIVE Started ---')

    while True:
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'FETUSDT']
        max_score = 0
        best_symbol = ""

        for symbol in symbols:
            # Mocking data for demonstration
            mock_row = pd.Series({
                'ret_1': 0.01, 'ret_4': 0.02, 'ret_12': 0.05, 'ret_24': 0.10,
                'dist_ma_12': 0.02, 'dist_ma_48': 0.05, 'rsi14': 65, 'atr_ratio': 0.005,
                'rel_strength': 0.02, 'trend_1h': True, 'close': 65000.0 if 'BTC' in symbol else 100.0
            })

            side, score = strategy.get_signal(symbol, mock_row)

            if score > max_score:
                max_score = score
                best_symbol = symbol

            if side:
                tp, sl = strategy.get_tp_sl(symbol, side, mock_row['close'], mock_row['atr_ratio'])
                slip = strategy.get_slippage(symbol)
                track = "1H" if symbol in ['BTCUSDT', 'ETHUSDT'] else "15m"
                msg = f'[{symbol} ({track})] | {side} | AI: {score:.4f} | Slip: {slip}bps | Price: {mock_row["close"]:.2f} | TP: {tp:.2f} | SL: {sl:.2f}'
                send_telegram(msg)
                strategy.record_exit(symbol)

        # Hourly Heartbeat (Snapshot 93.1 style)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        heartbeat = (
            f'\n--- [SYSTEM SNAPSHOT {now}] ---\n'
            f'Equity: $1000.00 | MaxDD: 0.00% | Rating: BULLISH\n'
            f'Best Candidate: {best_symbol} (AI: {max_score:.4f})\n'
            f'Active Tracks: BTC/ETH (1H), SOL/FET/AVAX (15m)\n'
            f'----------------------------------'
        )
        print(heartbeat)

        time.sleep(3600)

if __name__ == "__main__":
    run_production_cycle()
