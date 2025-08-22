import numpy as np
import pandas as pd
from typing import Dict


def summarize(equity_curve: pd.DataFrame, trades: pd.DataFrame, bar_seconds: int = 900) -> Dict[str, float]:
    """Summarize backtest performance metrics.

    Parameters
    ----------
    equity_curve : pd.DataFrame
        DataFrame containing equity progression with at least ``equity_before`` and
        ``equity_after`` columns. ``timestamp`` is optional but used for period
        calculations.
    trades : pd.DataFrame
        Executed trade records with columns such as ``pnl`` and ``bars_held``.
    bar_seconds : int, default 900
        Number of seconds per bar. 15 minute bars correspond to 900 seconds.

    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    metrics = {
        "total_return": 0.0,
        "annual_return": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "signal_count": 0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "avg_holding_minutes": 0.0,
        "exposure": 0.0,
        "pnl_std": 0.0,
    }

    eq_df = equity_curve.copy() if isinstance(equity_curve, pd.DataFrame) else pd.DataFrame()
    trades_df = trades.copy() if isinstance(trades, pd.DataFrame) else pd.DataFrame()

    if eq_df.empty and trades_df.empty:
        return metrics

    # --- Total return ---
    if not eq_df.empty:
        initial_eq = float(eq_df.get("equity_before", pd.Series([1.0])).iloc[0])
        final_eq = float(eq_df.get("equity_after", pd.Series([initial_eq])).iloc[-1])
    else:
        initial_eq = 1.0
        final_eq = 1.0
    total_return = (final_eq - initial_eq) / initial_eq if initial_eq != 0 else 0.0
    metrics["total_return"] = float(total_return)

    # Determine start/end time for annualization and exposure
    start_time = end_time = None
    if not trades_df.empty:
        start_time = pd.to_datetime(trades_df["entry_time"]).min()
        end_time = pd.to_datetime(trades_df["exit_time"]).max()
    elif not eq_df.empty and "timestamp" in eq_df.columns:
        start_time = pd.to_datetime(eq_df["timestamp"]).min()
        end_time = pd.to_datetime(eq_df["timestamp"]).max()

    # --- Annualized return ---
    if start_time is not None and end_time is not None and end_time > start_time:
        duration_seconds = (end_time - start_time).total_seconds()
        years = duration_seconds / (365.25 * 24 * 3600)
        if years > 0:
            metrics["annual_return"] = float((1 + total_return) ** (1 / years) - 1)

    # --- Max drawdown ---
    if not eq_df.empty and "equity_after" in eq_df.columns:
        equity = eq_df["equity_after"].astype(float)
        running_max = equity.cummax()
        drawdown = (equity / running_max) - 1.0
        if not drawdown.empty:
            metrics["max_drawdown"] = float(abs(drawdown.min()))

    # --- Sharpe ratio (per bar) ---
    if not eq_df.empty and len(eq_df) > 1 and "equity_after" in eq_df.columns:
        returns = eq_df["equity_after"].pct_change().dropna()
        if not returns.empty and returns.std() != 0:
            periods_per_year = 365.25 * 24 * 3600 / float(bar_seconds)
            metrics["sharpe_ratio"] = float(returns.mean() / returns.std() * np.sqrt(periods_per_year))

    # --- Trade based metrics ---
    if not trades_df.empty:
        metrics["signal_count"] = int(len(trades_df))
        wins = trades_df[trades_df["pnl"] > 0]["pnl"]
        losses = trades_df[trades_df["pnl"] < 0]["pnl"]
        metrics["win_rate"] = float(len(wins) / len(trades_df)) if len(trades_df) else 0.0
        metrics["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        metrics["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
        loss_sum = float(losses.sum())
        win_sum = float(wins.sum())
        metrics["profit_factor"] = float(win_sum / abs(loss_sum)) if loss_sum != 0 else (float("inf") if win_sum > 0 else 0.0)
        metrics["avg_holding_minutes"] = float(trades_df["bars_held"].mean() * (bar_seconds / 60.0)) if len(trades_df) else 0.0
        metrics["pnl_std"] = float(trades_df["pnl"].std(ddof=0)) if len(trades_df) else 0.0

        if start_time is not None and end_time is not None and end_time > start_time:
            total_seconds = (end_time - start_time).total_seconds()
            total_bars = total_seconds / float(bar_seconds)
            held_bars = float(trades_df["bars_held"].sum())
            metrics["exposure"] = float(held_bars / total_bars) if total_bars > 0 else 0.0

    return metrics
