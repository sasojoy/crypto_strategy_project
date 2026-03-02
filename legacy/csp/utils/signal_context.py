from dataclasses import dataclass
from typing import Optional, Dict
import math


@dataclass
class SignalContext:
    side: str               # "LONG" | "SHORT" | "NONE"
    score: float            # 模型分數 (0~1)
    threshold: float        # 決策門檻
    h_bars: int             # 固定持有幾根 (e.g., 16)
    pt: float               # 目標% (小數) 例: 0.008 代表 +0.8%
    sl: float               # 停損% (小數)
    entry_price: float
    up_price: Optional[float]  # 目標價
    down_price: Optional[float]# 停損價
    reason: str


def compute_risk_params(entry_price: float, atr: Optional[float], cfg: Dict) -> (float, float, float, float):
    """
    用 ATR 估目標/停損；若 ATR 不足，用 fallback 百分比。
    cfg 需要:
      - risk.atr_n (int): ATR 期數
      - risk.pt_atr_mult (float): 目標倍數
      - risk.sl_atr_mult (float): 停損倍數
      - risk.fallback_pt_pct (float): Fallback 目標百分比（小數）
      - risk.fallback_sl_pct (float): Fallback 停損百分比（小數）
    """
    pt_pct = cfg.get("risk", {}).get("fallback_pt_pct", 0.008)  # 0.8%
    sl_pct = cfg.get("risk", {}).get("fallback_sl_pct", 0.005)  # 0.5%
    if atr and atr > 0:
        pt_pct = atr * float(cfg.get("risk", {}).get("pt_atr_mult", 1.0)) / entry_price
        sl_pct = atr * float(cfg.get("risk", {}).get("sl_atr_mult", 0.6)) / entry_price
    up_price = entry_price * (1.0 + pt_pct)
    down_price = entry_price * (1.0 - sl_pct)
    return pt_pct, sl_pct, up_price, down_price


def build_signal_context(
    symbol: str,
    score: float,
    entry_price: float,
    horizon_bars: int,
    threshold: float,
    atr_value: Optional[float],
    filters: Dict,
    cfg: Dict
) -> SignalContext:
    """
    依 score 與 threshold 決定 side，並補齊 h/pt/↑/↓/reason。
    filters 可包含:
      - cooldown_pass (bool)
      - dd_guard_pass (bool)
      - session_pass (bool)
      - vol_pass (bool)
      - extra_reasons (list[str])
    """
    side = "NONE"
    reason_bits = []
    if score >= threshold:
        side = "LONG"
        reason_bits.append(f"proba≥thr ({score:.3f}≥{threshold:.2f})")
    elif score <= (1.0 - threshold):
        side = "SHORT"
        reason_bits.append(f"proba≤1-thr ({score:.3f}≤{1.0-threshold:.2f})")
    else:
        reason_bits.append(f"hold: score {score:.3f} in ({1.0-threshold:.2f},{threshold:.2f})")

    # 風控過濾器
    def tag(ok, name):
        reason_bits.append(f"{name}={'ok' if ok else 'block'}")
        return ok

    f_cd = tag(filters.get("cooldown_pass", True), "cooldown")
    f_dd = tag(filters.get("dd_guard_pass", True), "dd")
    f_sess = tag(filters.get("session_pass", True), "session")
    f_vol = tag(filters.get("vol_pass", True), "vol")
    if filters.get("extra_reasons"):
        reason_bits.extend(filters["extra_reasons"])

    # 若任何過濾器阻擋，則不進場
    if side != "NONE" and not all([f_cd, f_dd, f_sess, f_vol]):
        side = "NONE"

    pt_pct, sl_pct, up_px, dn_px = compute_risk_params(entry_price, atr_value, cfg)

    return SignalContext(
        side=side,
        score=score,
        threshold=threshold,
        h_bars=int(horizon_bars),
        pt=pt_pct,
        sl=sl_pct,
        entry_price=entry_price,
        up_price=up_px if side=="LONG" else (entry_price*(1.0 - pt_pct) if side=="SHORT" else None),
        down_price=dn_px if side=="LONG" else (entry_price*(1.0 + sl_pct) if side=="SHORT" else None),
        reason="; ".join(reason_bits)
    )

