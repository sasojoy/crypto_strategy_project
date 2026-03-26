import pandas as pd
import numpy as np
import joblib
import pickle
import os
from datetime import datetime, timedelta

class TradingStrategy:
    def __init__(self):
        # Models
        self.core_model_path = 'h16_dynamic/model_h1_v111.pkl'
        self.core_scaler_path = 'h16_dynamic/scaler_h1_v111.joblib'
        self.alt_model_path = 'h16_dynamic/model_alt_v113.pkl'
        self.alt_scaler_path = 'h16_dynamic/scaler_alt_v113.joblib'

        with open(self.core_model_path, 'rb') as f: self.core_model = pickle.load(f)
        self.core_scaler = joblib.load(self.core_scaler_path)
        with open(self.alt_model_path, 'rb') as f: self.alt_model = pickle.load(f)
        self.alt_scaler = joblib.load(self.alt_scaler_path)

        # Feature Lists
        self.core_features = ['ret_1', 'ret_4', 'ret_12', 'ret_24', 'dist_ma_12', 'dist_ma_48', 'rsi14', 'atr_ratio']
        self.alt_features = ['ret_1', 'ret_4', 'ret_12', 'rsi14', 'atr_ratio', 'rel_strength']

        # Config
        self.max_total_equity = 1000.0
        self.cooldown_hours = 4
        self.last_exit_time = {} # {symbol: datetime}

        self.slippage_map = {
            'BTCUSDT': 5, 'ETHUSDT': 5, 'SOLUSDT': 15, 'AVAXUSDT': 15, 'FETUSDT': 30
        }

    def get_signal(self, symbol, row):
        if symbol in self.last_exit_time:
            if datetime.now() < self.last_exit_time[symbol] + timedelta(hours=self.cooldown_hours):
                return None, 0

        is_core = symbol in ['BTCUSDT', 'ETHUSDT', 'BTC/USDT', 'ETH/USDT']
        model = self.core_model if is_core else self.alt_model
        scaler = self.core_scaler if is_core else self.alt_scaler
        features = self.core_features if is_core else self.alt_features
        threshold = 0.85 if is_core else 0.75

        X_df = pd.DataFrame([row[features].values], columns=features)
        probs = model.predict_proba(scaler.transform(X_df))[0]

        if is_core:
            p_long, p_short = probs[1], probs[2]
            if p_long >= threshold: return 'LONG', p_long
            if p_short >= threshold: return 'SHORT', p_short
            return None, max(p_long, p_short)
        else:
            p_long = probs[1]
            if p_long >= threshold and row.get('trend_1h', True):
                return 'LONG', p_long
            return None, p_long

    def get_tp_sl(self, symbol, side, price, atr_ratio):
        is_core = symbol in ['BTCUSDT', 'ETHUSDT', 'BTC/USDT', 'ETH/USDT']
        atr = price * atr_ratio
        if is_core:
            tp = price * (1.05 if side == 'LONG' else 0.95)
            sl = price - 1.5 * atr if side == 'LONG' else price + 1.5 * atr
        else:
            base_tp = 0.03
            slip_comp = 0.005 if self.slippage_map.get(symbol, 30) >= 30 else 0
            tp = price * (1 + base_tp + slip_comp) if side == 'LONG' else price * (1 - (base_tp + slip_comp))
            sl = price - 1.5 * atr if side == 'LONG' else price + 1.5 * atr
        return tp, sl

    def get_slippage(self, symbol):
        return self.slippage_map.get(symbol, 30)

    def record_exit(self, symbol):
        self.last_exit_time[symbol] = datetime.now()
