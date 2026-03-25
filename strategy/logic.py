
import os
import json
import hashlib
import numpy as np
import pandas as pd

class TradingStrategy:
    """
    Iteration 105.1: Standardized Trading Strategy Logic
    Encapsulates entry/exit rules for both backtest and live trading.
    """
    def __init__(self, config_path=None, th_base=0.92, tp_pct=0.025, sl_pct=0.010):
        self.config = {}
        self.config_hash = "default"
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                content = f.read()
                self.config_hash = hashlib.md5(content.encode()).hexdigest()
                f.seek(0)
                try:
                    self.config = json.load(f)
                except:
                    self.config = {}
        
        self.th_base = th_base
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.adx_threshold = 30

    @property
    def min_hold_minutes(self):
        return 30

    def get_thresholds(self, current_price, ema_200, atr_ratio, btc_falling=False):
        # Iteration 105.1: High Conviction
        th_long = self.th_base
        th_short = self.th_base
        return th_long, th_short

    def check_entry(self, up_prob, dn_prob, features, regime):
        current_price = features.get('close', 0)
        ema_200 = features.get('ema_200', current_price)
        atr_ratio = features.get('atr_ratio', 0)
        btc_change_5m = features.get('btc_change_5m', 0.0)
        btc_falling = btc_change_5m < 0
        
        # ADX Filter
        adx = features.get('adx14', 0)
        if adx < self.adx_threshold:
            return None, 1.0

        th_l, th_s = self.get_thresholds(current_price, ema_200, atr_ratio, btc_falling)

        # Basic filters
        ema_alignment_long = (current_price > features.get('ema_slow', 0))
        ema_alignment_short = (current_price < features.get('ema_slow', 0))
        
        long_ok = (up_prob >= th_l) and regime.get("regime_long_ok", True) and ema_alignment_long
        short_ok = (dn_prob >= th_s) and regime.get("regime_short_ok", True) and ema_alignment_short

        if long_ok: return "LONG", 1.0
        if short_ok: return "SHORT", 1.0
        return None, 1.0

    def build_tp_sl(self, side, entry_price):
        if side == "LONG":
            sl = round(entry_price * (1 - self.sl_pct), 2)
            tp = round(entry_price * (1 + self.tp_pct), 2)
        else:
            sl = round(entry_price * (1 + self.sl_pct), 2)
            tp = round(entry_price * (1 - self.tp_pct), 2)
        return sl, tp

def compute_adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = pd.to_numeric(df["high"], errors="coerce").ffill()
    l = pd.to_numeric(df["low"], errors="coerce").ffill()
    c = pd.to_numeric(df["close"], errors="coerce").ffill()
    up_move   = np.r_[np.nan, np.diff(h)]
    down_move = -np.r_[np.nan, np.diff(l)]
    plus_dm   = np.where((up_move > down_move) & (up_move > 0),  up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    pc = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - pc).abs(),
        (l - pc).abs()
    ], axis=1).max(axis=1)
    
    def _rma(s, n):
        return s.ewm(alpha=1/n, adjust=False).mean()
        
    atr = _rma(tr, n)
    plus_di  = 100 * _rma(pd.Series(plus_dm, index=df.index), n) / (atr + 1e-12)
    minus_di = 100 * _rma(pd.Series(minus_dm, index=df.index), n) / (atr + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = _rma(dx, n)
    return adx
