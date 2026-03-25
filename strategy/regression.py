import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import yaml
from datetime import datetime
import pandas as pd
import numpy as np

def load_current_position(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_current_position(position, path):
    # 自動轉換為 YAML-safe 類型
    for k, v in list(position.items()):
        if isinstance(v, np.generic):
            position[k] = v.item()
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(position, f)

def clear_position(path):
    if os.path.exists(path):
        os.remove(path)

def _compute_holding_minutes(position):
    """
    若 position 內已累計 bars_held（以根數為單位），優先使用以避免時區/系統時間差；
    否則回退用 entry_time 與當下時間計算。
    """
    bars_held = position.get("bars_held")
    if isinstance(bars_held, (int, float)) and bars_held >= 0:
        return int(bars_held) * 15

    entry_time = datetime.strptime(position["entry_time"], "%Y-%m-%d %H:%M:%S")
    return int((datetime.now() - entry_time).total_seconds() / 60)

def check_exit_condition(position, current_price):
    """
    統一出口條件：
    1) 先檢查 TP/SL（若 position 內有帶 tp/sl）
    2) 再檢查原有固定條件（+5%／-2%／最長持倉）
    備註：最長持倉可由 position["max_hold_bars"] 覆蓋（單位：根），否則預設 240 分鐘。
    """
    entry_price = float(position["entry_price"])
    side = position.get("side", "LONG")
    holding_minutes = _compute_holding_minutes(position)
    
    # Iteration 101.0: Minimum Hold Time Protection (30 minutes)
    # Unless it's a hard SL, we don't allow TP before 30 minutes.
    min_hold_minutes = 30

    # --- 1) 先檢查 TP / SL（若存在） ---
    sl = position.get("sl")
    tp = position.get("tp")
    
    # Iteration 102.0: Dynamic Trailing Stop-Loss (1.0% trail after +2.0% profit)
    # CEO Requirement: Capture big moves (5%+) while protecting gains.
    unrealized_pnl = (current_price - entry_price) / entry_price if side == "LONG" else (entry_price - current_price) / entry_price
    
    # Track High-Water Mark (HWM) for trailing
    hwm = position.get("hwm", entry_price)
    if side == "LONG":
        hwm = max(hwm, current_price)
    else:
        hwm = min(hwm, current_price)
    position["hwm"] = hwm

    # Iteration 104.0: Scale-out Profit Taking & Trailing Stop
    if unrealized_pnl >= 0.030:
        # 獲利達 +3.0% 時：啟動 Trailing Stop (1.5% 追蹤)
        trail_pct = 0.015
        if side == "LONG":
            new_sl = hwm * (1 - trail_pct)
            sl = max(sl, new_sl) if sl is not None else new_sl
        else:
            new_sl = hwm * (1 + trail_pct)
            sl = min(sl, new_sl) if sl is not None else new_sl
        position["sl"] = sl
    elif unrealized_pnl >= 0.015:
        # 獲利達 +1.5% 時：平倉 50% (Scale-out)
        # Note: In this simplified logic, we mark it as "SCALED_OUT" in position
        if not position.get("scaled_out", False):
            position["scaled_out"] = True
            # Move SL to Entry + 0.5%
            if side == "LONG":
                sl = entry_price * 1.005
            else:
                sl = entry_price * 0.995
            position["sl"] = sl
            # We don't exit here, but we've "secured" half profit conceptually
            # In a real exchange, we would send a partial close order.

    if sl is not None and tp is not None:
        if side == "LONG":
            if current_price >= tp:
                if holding_minutes >= min_hold_minutes:
                    return {"exit": True, "reason": "TP", "holding_minutes": holding_minutes}
            if current_price <= sl:
                reason = "BREAKEVEN" if sl > entry_price else "SL"
                return {"exit": True, "reason": reason, "holding_minutes": holding_minutes}
        else:  # SHORT
            if current_price <= tp:
                if holding_minutes >= min_hold_minutes:
                    return {"exit": True, "reason": "TP", "holding_minutes": holding_minutes}
            if current_price >= sl:
                reason = "BREAKEVEN" if sl < entry_price else "SL"
                return {"exit": True, "reason": reason, "holding_minutes": holding_minutes}

    # --- 2) 原有固定條件（回溯相容） ---
    if side == "SHORT":
        return_pct = (entry_price - current_price) / entry_price
    else:
        return_pct = (current_price - entry_price) / entry_price

    # Iteration 104.0: Time-based Exit
    # 如果持倉超過 24 小時 (1440 min) 且利潤低於 0.5%，強制平倉換手
    if holding_minutes >= 1440 and return_pct < 0.005:
        return {"exit": True, "reason": "TIME_EXIT_LOW_PNL", "holding_minutes": holding_minutes}

    # 固定止盈 / 止損（相容以前的 +5% / -2%）
    if return_pct >= 0.05:
        return {"exit": True, "reason": "達到止盈", "holding_minutes": holding_minutes}
    elif return_pct <= -0.02:
        return {"exit": True, "reason": "觸發止損", "holding_minutes": holding_minutes}

    # 最長持倉
    max_hold_bars = position.get("max_hold_bars")
    if isinstance(max_hold_bars, (int, float)) and max_hold_bars > 0:
        if holding_minutes > int(max_hold_bars) * 15:
            return {"exit": True, "reason": "超時出場", "holding_minutes": holding_minutes}
    else:
        if holding_minutes > 240:
            # Default for cls_h16 is 16 bars = 240 min, but we keep it for safety
            return {"exit": True, "reason": "超時出場", "holding_minutes": holding_minutes}

    # 持續持有
    return {"exit": False, "holding_minutes": holding_minutes}

def log_trade(trade, log_path="resources/trade_log.csv"):
    """
    直接將 trade 字典落盤；若缺少欄位也不會出錯。
    建議在呼叫端（realtime_cls）把以下欄位一併寫入 trade，方便日後分析：
      - tp_sl_mode, sl, tp, sl_mult, tp_mult, atr_at_entry, bars_held
    """
    df = pd.DataFrame([trade])
    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)
