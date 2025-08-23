from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Return percentile rank of the last value within a rolling window."""

    def percentile_of_last(values: np.ndarray) -> float:
        arr = np.sort(values)
        # percentile rank (0-1) of the last element
        return np.searchsorted(arr, values[-1], side="right") / len(arr)

    return series.rolling(window).apply(percentile_of_last, raw=True)


def add_features(
    df: pd.DataFrame,
    *,
    prev_high_period: int = 20,
    prev_low_period: int = 20,
    bb_window: int = 20,
    atr_window: int = 14,
    atr_percentile_window: int = 100,
) -> pd.DataFrame:
    """Add additional technical features required by classifiers.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``open``, ``high``, ``low``, ``close`` and ``volume``.
    prev_high_period, prev_low_period : int
        Lookback periods for previous high/low distance features.
    bb_window : int
        Window for Bollinger Bands if not already present.
    atr_window : int
        Window for ATR calculation.
    atr_percentile_window : int
        Lookback for ATR percentile rank.
    """
    out = df.copy()

    # Previous high/low distance
    rolling_high = out["high"].rolling(prev_high_period).max()
    rolling_low = out["low"].rolling(prev_low_period).min()
    out["prev_high_dist"] = (out["high"] - rolling_high) / rolling_high
    out["prev_low_dist"] = (out["low"] - rolling_low) / rolling_low

    # Candle shape
    out["candle_body"] = out["close"] - out["open"]
    out["candle_upper"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["candle_lower"] = out[["open", "close"]].min(axis=1) - out["low"]

    body = out["candle_body"]
    prev_body = body.shift(1)
    curr_high_body = out[["open", "close"]].max(axis=1)
    curr_low_body = out[["open", "close"]].min(axis=1)
    prev_high_body = out[["open", "close"]].shift(1).max(axis=1)
    prev_low_body = out[["open", "close"]].shift(1).min(axis=1)

    engulf_long = (
        (body.abs() > prev_body.abs())
        & (curr_low_body <= prev_low_body)
        & (curr_high_body >= prev_high_body)
        & (body > 0)
        & (prev_body < 0)
    )
    engulf_short = (
        (body.abs() > prev_body.abs())
        & (curr_low_body <= prev_low_body)
        & (curr_high_body >= prev_high_body)
        & (body < 0)
        & (prev_body > 0)
    )
    engulf = pd.Series(0, index=out.index, dtype=int)
    engulf[engulf_long] = 1
    engulf[engulf_short] = -1
    out["candle_engulfing"] = engulf

    # Bollinger Band width
    if {"bb_high", "bb_low", "bb_mid"}.issubset(out.columns):
        upper = out["bb_high"]
        lower = out["bb_low"]
        mid = out["bb_mid"]
    else:
        m = out["close"].rolling(bb_window).mean()
        s = out["close"].rolling(bb_window).std()
        mid = m
        upper = m + 2 * s
        lower = m - 2 * s
    out["bb_width"] = (upper - lower) / mid

    # VWAP deviation
    typical_price = (out["high"] + out["low"] + out["close"]) / 3
    cum_vol = out["volume"].cumsum()
    cum_vp = (typical_price * out["volume"]).cumsum()
    vwap = cum_vp / cum_vol.replace(0, np.nan)
    out["vwap_dev"] = (out["close"] - vwap) / vwap

    # ATR and its percentile
    h = out["high"]
    l = out["low"]
    c = out["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_window).mean()
    out["atr_percentile"] = _rolling_percentile(atr, atr_percentile_window)

    out = out.dropna().reset_index(drop=True)
    return out
