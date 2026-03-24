






import sys, os
import pytest
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.realtime_cls import TradingStrategy

def test_strategy_consistency():
    """
    Smoke Test: Verify TradingStrategy logic consistency.
    """
    # Mock config
    config = {"best": {"th_long": 0.85, "th_short": 0.85}}
    config_path = "tests/mock_opt.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    
    try:
        strategy = TradingStrategy(config_path)
        
        # Mock data
        entry_price = 100.0
        
        # Test TP/SL
        sl_l, tp_l = strategy.build_tp_sl("LONG", entry_price)
        assert sl_l == 99.0  # 1.0% SL
        assert tp_l == 101.5 # 1.5% TP
        
        sl_s, tp_s = strategy.build_tp_sl("SHORT", entry_price)
        assert sl_s == 101.0 # 1.0% SL
        assert tp_s == 98.5  # 1.5% TP
        
        # Test Entry Logic (Normal Market)
        features = {
            'close': 100, 'ema_200': 100, 'atr_ratio': 0.01,
            'btc_change_5m': 0.0, 'volume': 200, 'vol_ma_24h': 100,
            'rsi_slope': 5.0, 'ema_slow': 90
        }
        regime = {"regime_long_ok": True, "regime_short_ok": True}
        
        side = strategy.check_entry(up_prob=0.95, dn_prob=0.05, features=features, regime=regime)
        assert side == "LONG"
        
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == "__main__":
    test_strategy_consistency()






