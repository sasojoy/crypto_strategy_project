import pytest
import pandas as pd
import numpy as np
import os, sys, warnings
sys.path.append(os.getcwd())
from strategy.logic import TradingStrategy

def test_no_warnings():
    strategy = TradingStrategy()
    
    # Mock Core Row (BTC)
    core_row = pd.Series({
        'ret_1': 0.01, 'ret_4': 0.02, 'ret_12': 0.05, 'ret_24': 0.10,
        'dist_ma_12': 0.02, 'dist_ma_48': 0.05, 'rsi14': 65, 'atr_ratio': 0.005
    })
    
    # Mock Alt Row (SOL)
    alt_row = pd.Series({
        'ret_1': 0.01, 'ret_4': 0.02, 'ret_12': 0.05, 'rsi14': 65, 
        'atr_ratio': 0.01, 'rel_strength': 0.02, 'trend_1h': True
    })
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        
        # Test Core
        strategy.get_signal('BTCUSDT', core_row)
        # Test Alt
        strategy.get_signal('SOLUSDT', alt_row)
        
        # Filter for UserWarnings from sklearn/joblib
        user_warnings = [warning for warning in w if issubclass(warning.category, UserWarning)]
        assert len(user_warnings) == 0, f'Found {len(user_warnings)} UserWarnings: {[str(uw.message) for uw in user_warnings]}'

if __name__ == "__main__":
    pytest.main([__file__])
