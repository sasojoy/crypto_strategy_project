from __future__ import annotations
import pandas as pd


def make_labels(df: pd.DataFrame, horizons: list[int], thresholds: list[float]) -> pd.DataFrame:
    """Generate multi-horizon, multi-threshold binary labels.

    Parameters
    ----------
    df : DataFrame
        Must contain a ``close`` column and be indexed by time.
    horizons : list[int]
        Number of future bars to look ahead for each horizon.
    thresholds : list[float]
        Thresholds in percent. 0.2 means future return >0.2%.

    Returns
    -------
    DataFrame
        Columns have MultiIndex (horizon, threshold) with values 0 or 1.
        Last ``max(horizons)`` rows are dropped due to shift.
    """
    if "close" not in df.columns:
        raise KeyError("DataFrame must contain 'close' column")

    close = df["close"].astype(float)
    labels = {}
    for h in horizons:
        future = close.shift(-h)
        ret = (future / close - 1.0) * 100.0
        for t in thresholds:
            labels[(h, t)] = (ret > t).astype(int)
    df_labels = pd.DataFrame(labels, index=df.index)
    df_labels.columns = pd.MultiIndex.from_tuples(df_labels.columns, names=["horizon", "threshold"])
    df_labels = df_labels.dropna()
    return df_labels
