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

    # --- 1) 先檢查 TP / SL（若存在） ---
    sl = position.get("sl")
    tp = position.get("tp")
    if sl is not None and tp is not None:
        if side == "LONG":
            if current_price >= tp:
                return {"exit": True, "reason": "TP", "holding_minutes": holding_minutes}
            if current_price <= sl:
                return {"exit": True, "reason": "SL", "holding_minutes": holding_minutes}
        else:  # SHORT
            if current_price <= tp:
                return {"exit": True, "reason": "TP", "holding_minutes": holding_minutes}
            if current_price >= sl:
                return {"exit": True, "reason": "SL", "holding_minutes": holding_minutes}

    # --- 2) 原有固定條件（回溯相容） ---
    if side == "SHORT":
        return_pct = (entry_price - current_price) / entry_price
    else:
        return_pct = (current_price - entry_price) / entry_price

    # 固定止盈 / 止損（相容以前的 +5% / -2%）
    if return_pct >= 0.05:
        return {"exit": True, "reason": "達到止盈", "holding_minutes": holding_minutes}
    elif return_pct <= -0.02:
        return {"exit": True, "reason": "觸發止損", "holding_minutes": holding_minutes}

    # 最長持倉：優先讀 position["max_hold_bars"]，否則 240 分鐘
    max_hold_bars = position.get("max_hold_bars")
    if isinstance(max_hold_bars, (int, float)) and max_hold_bars > 0:
        if holding_minutes > int(max_hold_bars) * 15:
            return {"exit": True, "reason": "超時出場", "holding_minutes": holding_minutes}
    else:
        if holding_minutes > 240:
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
