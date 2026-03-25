





import sys, os
import pytest
import hashlib
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.realtime_cls import TradingStrategy

@pytest.fixture
def strategy(tmp_path):
    config = {"best": {"th_long": 0.85, "th_short": 0.85}}
    config_file = tmp_path / "opt_h16_dynamic.json"
    config_file.write_text(json.dumps(config))
    return TradingStrategy(str(config_file))

def test_tp_sl_calculation(strategy):
    entry_price = 100.0
    sl_long, tp_long = strategy.build_tp_sl("LONG", entry_price)
    assert sl_long == 99.0  # 1.0% SL
    assert tp_long == 101.5 # 1.5% TP
    
    sl_short, tp_short = strategy.build_tp_sl("SHORT", entry_price)
    assert sl_short == 101.0 # 1.0% SL
    assert tp_short == 98.5  # 1.5% TP

def test_threshold_asymmetry(strategy):
    # Trend Down (Price < EMA200)
    th_l, th_s = strategy.get_thresholds(current_price=90, ema_200=100, atr_ratio=0.01)
    assert th_l == pytest.approx(0.90) # 0.85 + 0.05
    assert th_s == pytest.approx(0.80) # 0.85 - 0.05
    
    # Trend Up (Price > EMA200)
    th_l, th_s = strategy.get_thresholds(current_price=110, ema_200=100, atr_ratio=0.01)
    assert th_l == pytest.approx(0.85)
    assert th_s == pytest.approx(0.85)

def test_volatility_compensation(strategy):
    # Quiet Market (ATR Ratio < 0.005) -> Threshold + 0.05
    th_l, th_s = strategy.get_thresholds(current_price=100, ema_200=100, atr_ratio=0.004)
    assert th_l == pytest.approx(0.90) # 0.85 + 0.05
    assert th_s == pytest.approx(0.90) # 0.85 + 0.05
    
    # Noisy Market (ATR Ratio > 0.015) -> Threshold + 0.05
    th_l, th_s = strategy.get_thresholds(current_price=100, ema_200=100, atr_ratio=0.016)
    assert th_l == pytest.approx(0.90)
    assert th_s == pytest.approx(0.90)

def test_btc_correlation_filter(strategy):
    features = {
        'close': 100, 'ema_200': 100, 'atr_ratio': 0.01,
        'btc_change_5m': -0.015, # BTC Crashing
        'volume': 100, 'vol_ma_24h': 100, 'rsi_slope': 5.0, 'ema_slow': 90
    }
    regime = {"regime_long_ok": True, "regime_short_ok": True}
    
    # Should block LONG even if prob is high
    side, size = strategy.check_entry(up_prob=0.95, dn_prob=0.1, features=features, regime=regime)
    assert side is None

def test_panic_indicator(strategy):
    features = {
        'close': 80, 'ema_200': 100, 'atr_ratio': 0.01,
        'btc_change_5m': -0.001, # BTC falling
        'volume': 400, 'vol_ma_24h': 100, # 4x Volume
        'bb_lower': 85, # Price below BB lower
        'rsi_slope': -5.0, 'ema_slow': 110
    }
    regime = {"regime_long_ok": True, "regime_short_ok": True}
    
    # Should trigger SHORT even if prob is low
    side, size = strategy.check_entry(up_prob=0.1, dn_prob=0.5, features=features, regime=regime)
    assert side == "SHORT"
    assert size == 1.2 # Short Assault


def test_partial_fill_logic(strategy):
    # Iteration 104.0: Scale-out TP (50% at +1.5%)
    # This is handled in the backtest loop, but we can test the tighten_stop_only logic
    from pipeline.backtest_cls_atr_dynamic import tighten_stop_only
    
    entry_price = 100.0
    current_sl = 98.0
    atr_now = 1.0
    
    # Long: tighten stop to entry - 1.2 * ATR
    new_sl = tighten_stop_only("LONG", current_sl, entry_price, atr_now)
    assert new_sl == 98.8 # max(98.0, 100.0 - 1.2)
    
    # Short: tighten stop to entry + 1.2 * ATR
    current_sl_short = 102.0
    new_sl_short = tighten_stop_only("SHORT", current_sl_short, entry_price, atr_now)
    assert new_sl_short == 101.2 # min(102.0, 100.0 + 1.2)






