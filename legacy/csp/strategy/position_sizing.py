from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class ExchangeRule:
    """Simple representation of exchange trading rules.

    Attributes
    ----------
    min_qty : float
        Minimum order quantity of the base asset.
    qty_step : float
        Quantity increment step. Order quantity must be a multiple of
        ``qty_step``.
    min_notional : float
        Minimum notional value (``qty * price``) allowed.
    max_leverage : int
        Maximum leverage supported by the exchange. Not used directly in
        sizing but kept for future extension.
    """
    min_qty: float
    qty_step: float
    min_notional: float
    max_leverage: int

@dataclass
class SizingInput:
    """Input parameters for position sizing calculations."""
    equity_usdt: float
    entry_price: float
    atr_abs: float
    side: Literal["LONG", "SHORT"]
    tp_ratio: float
    sl_ratio: float
    win_rate: Optional[float] = None
    rule: Optional[ExchangeRule] = None


def kelly_fraction(win_rate: float, r_multiple: float) -> float:
    """Return the Kelly optimal fraction ``f*``.

    Parameters
    ----------
    win_rate : float
        Historical win probability ``p`` in the range ``[0, 1]``.
    r_multiple : float
        Average win/loss ratio ``R``. For example ``tp_ratio/sl_ratio``.

    Returns
    -------
    float
        The Kelly fraction clipped to ``[0, 1]``. Returns ``0`` when the
        inputs are invalid (non-positive ``R`` or ``p`` outside ``[0,1]``).
    """
    try:
        p = float(win_rate)
        R = float(r_multiple)
    except Exception:
        return 0.0
    if not (0.0 <= p <= 1.0) or R <= 0:
        return 0.0
    f = p - (1 - p) / R
    if f < 0:
        return 0.0
    return float(max(0.0, min(f, 1.0)))


def atr_position_size(equity_usdt: float, atr_abs: float, entry_price: float,
                      risk_per_trade: float, atr_k: float) -> float:
    """Position size based on ATR risk equalisation.

    The position size is determined so that the dollar risk per trade does not
    exceed ``equity_usdt * risk_per_trade``.

    Parameters
    ----------
    equity_usdt : float
        Account equity in USDT.
    atr_abs : float
        Absolute ATR value in price units.
    entry_price : float
        Intended entry price.
    risk_per_trade : float
        Maximum fraction of equity to risk per trade.
    atr_k : float
        Multiplier of ATR used to approximate stop distance.

    Returns
    -------
    float
        Quantity of base asset. Returns ``0`` if inputs are invalid or ATR is
        non-positive.
    """
    if atr_abs <= 0 or entry_price <= 0 or equity_usdt <= 0:
        return 0.0
    risk_usd = equity_usdt * risk_per_trade
    denom = atr_abs * atr_k * entry_price
    if denom <= 0:
        return 0.0
    qty = risk_usd / denom
    return float(max(qty, 0.0))


def apply_exchange_rule(qty: float, price: float, rule: Optional[ExchangeRule]) -> float:
    """Apply exchange constraints to ``qty``.

    The function enforces ``min_qty``, rounds the quantity to the nearest
    multiple of ``qty_step`` and checks the notional value ``qty*price``.

    Parameters
    ----------
    qty : float
        Proposed position size.
    price : float
        Latest price used for notional calculation.
    rule : ExchangeRule or ``None``
        Exchange trading rule. If ``None`` the quantity is returned as is.

    Returns
    -------
    float
        Adjusted quantity. Returns ``0`` if the order violates any rule.
    """
    if rule is None:
        return float(qty)
    if qty <= 0 or price <= 0:
        return 0.0
    if qty < rule.min_qty:
        return 0.0
    # Round to nearest step
    step = rule.qty_step
    if step > 0:
        qty = round(qty / step) * step
    notional = qty * price
    if notional < rule.min_notional:
        return 0.0
    return float(qty)


def blended_sizing(inp: SizingInput, mode: Literal["atr", "kelly", "hybrid"],
                   risk_per_trade: float, atr_k: float, kelly_coef: float = 0.5,
                   kelly_floor: float = -0.5, kelly_cap: float = 1.0) -> float:
    """Calculate position size using ATR, Kelly or a hybrid of both.

    Parameters
    ----------
    inp : SizingInput
        Core inputs including equity, price and ATR.
    mode : {"atr", "kelly", "hybrid"}
        Sizing mode.
    risk_per_trade : float
        Percentage of equity to risk per trade for ATR sizing.
    atr_k : float
        ATR multiplier for stop distance.
    kelly_coef : float, optional
        Weight of Kelly fraction in hybrid mode.
    kelly_floor : float, optional
        Minimum scaling factor minus one. ``-0.5`` means at most reduce to
        ``0.5`` of base size.
    kelly_cap : float, optional
        Maximum scaling factor minus one. ``1.0`` means at most double the
        base size.

    Returns
    -------
    float
        Final quantity after applying exchange rules. Returns ``0`` if the
        computed quantity is below exchange requirements.
    """
    ps_mode = mode.lower()
    rule = inp.rule

    base_qty = 0.0
    if ps_mode in ("atr", "hybrid"):
        base_qty = atr_position_size(inp.equity_usdt, inp.atr_abs,
                                     inp.entry_price, risk_per_trade, atr_k)
    if ps_mode == "kelly":
        if inp.win_rate is None:
            return 0.0
        f = kelly_fraction(inp.win_rate, inp.tp_ratio / inp.sl_ratio)
        qty = inp.equity_usdt * f / inp.entry_price
        return apply_exchange_rule(qty, inp.entry_price, rule)

    # hybrid or atr only
    if ps_mode == "atr" or base_qty <= 0:
        return apply_exchange_rule(base_qty, inp.entry_price, rule)

    # hybrid with Kelly adjustment
    kelly_f = 0.0
    if inp.win_rate is not None:
        r_mult = inp.tp_ratio / inp.sl_ratio if inp.sl_ratio > 0 else 0.0
        kelly_f = kelly_fraction(inp.win_rate, r_mult)
    scale = 1.0 + kelly_coef * kelly_f
    scale = max(1.0 + kelly_floor, min(scale, 1.0 + kelly_cap))
    qty = base_qty * scale
    return apply_exchange_rule(qty, inp.entry_price, rule)
