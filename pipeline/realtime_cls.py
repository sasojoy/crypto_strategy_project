# realtime_cls.py —— 15m + 4h 特徵、雙門檻、多空不對稱 ATR TP/SL（與訓練一致）
import sys, os, time, json, pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# 你現有專案的工具（沿用）
from util.notifier import notify
from strategy.regression import (
    load_current_position, save_current_position,
    clear_position, log_trade, check_exit_condition,
)
from pipeline.sync_to_sheets import sync_trade_to_sheet

# ===== 檔案路徑 =====
LIVE_DATA_PATH = "resources/live_data.csv"
MODEL_PATH     = "h16_dynamic/cls_model_h16.pkl"
SCALER_PATH    = "h16_dynamic/scaler_h16.joblib"
OPT_PATH       = "h16_dynamic/opt_h16_dynamic.json"
POSITION_PATH  = "resources/current_position.yaml"

# ===== 交易/持倉參數（與訓練一致）=====
H                 = 16              # 最多持倉 16 根（=4h）
ATR_N             = 14
MIN_HOLD_BARS     = 2               # 至少持有 2 根再檢查收緊
MAX_HOLD_BARS     = 16
FETCH_MIN_BARS    = 500             # 特徵需要較長歷史（跨週期RSI/斜率）
SYMBOL            = "BTCUSDT"

# ===== 訓練門檻與TP/SL（從 opt_h16_dynamic.json 讀）=====
with open(OPT_PATH, "r", encoding="utf-8") as f:
    _opt = json.load(f)["best"]
TH_LONG = float(_opt["th_long"])
TH_SHORT= float(_opt["th_short"])
TP_L    = float(_opt["tpL"])
SL_L    = float(_opt["slL"])
TP_S    = float(_opt["tpS"])
SL_S    = float(_opt["slS"])

# ===== 載入模型 & 標準化器 =====
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)
scaler = joblib.load(SCALER_PATH)

# ================== 資料取得 ==================
def _now_utc():
    return datetime.now(timezone.utc)

def fetch_klines_and_append_to_local(local_path=LIVE_DATA_PATH, need_bars=FETCH_MIN_BARS):
    """
    從 Binance 取得 15m K 線，補到本地 CSV；保留最後 need_bars 根
    """
    url = "https://api.binance.com/api/v3/klines"

    if os.path.exists(local_path):
        df_old = pd.read_csv(local_path)
        df_old["timestamp"] = pd.to_datetime(df_old["timestamp"], utc=True)
        last_time = df_old["timestamp"].max()
        # 預估還需要的根數（15m 一根）
        mins_gap = max(0, int((_now_utc() - last_time).total_seconds() // 900))
        need = max(mins_gap, 0)
    else:
        df_old = None
        need = need_bars

    if need > 0:
        params = {"symbol": SYMBOL, "interval": "15m", "limit": min(1000, need)}
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        arr = resp.json()
        df_new = pd.DataFrame(arr, columns=[
            "open_time","open","high","low","close","volume",
            "_1","_2","_3","_4","_5","_6"
        ])[["open_time","open","high","low","close","volume"]]
        df_new.rename(columns={"open_time":"timestamp"}, inplace=True)
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
        df_new = df_new.dropna()

        df_all = pd.concat([df_old, df_new], ignore_index=True) if df_old is not None else df_new
    else:
        df_all = df_old

    df_all = df_all.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    # 只保留最後 need_bars 根（至少 ensure）
    if len(df_all) > need_bars:
        df_all = df_all.iloc[-need_bars:].reset_index(drop=True)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df_all.to_csv(local_path, index=False)
    print(f"✅ 最新時間：{df_all['timestamp'].iloc[-1]}；共 {len(df_all)} 根 15m")
    return df_all

# ================== 特徵（與訓練一致，內建以減少依賴） ==================
def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100/(1+rs))

def _ema(s, span): return s.ewm(span=span, adjust=False).mean()

def _atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - pc).abs(),
        (df["low"]  - pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _slope_log(s, w):
    x = np.arange(w, dtype=float)
    def _f(win):
        if win.isna().any(): return np.nan
        y = np.log(win.values + 1e-12)
        vx = x - x.mean()
        return (vx * (y - y.mean())).sum() / (vx**2).sum()
    return s.rolling(w).apply(_f, raw=False) / (w + 1e-12)

def _zscore(s, w):
    m = s.rolling(w).mean()
    sd = s.rolling(w).std(ddof=0)
    return (s - m) / (sd + 1e-12)

FEATURES = [
    "ret_1","ret_4","ret_16","ret_32",
    "bb_z20","rv_16","slope_log_8","slope_log_16","slope_log_32",
    "ema_fast_dist","ema_fast_slow_gap","mom_ratio",
    "vol_chg","vol_z48","atr14","atr_ratio",
    "rsi14_15m","rsi14_1h","rsi14_4h",
]

def build_features_live(df15: pd.DataFrame) -> pd.DataFrame:
    """
    與訓練版等價，但不產生 y_up、不丟掉尾巴；回傳含 FEATURES 欄位的 DataFrame（最後一列用來預測）
    """
    df = df15.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    o,h,l,c,v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    df["ret_1"]  = c.pct_change(1)
    df["ret_4"]  = c.pct_change(4)
    df["ret_16"] = c.pct_change(16)
    df["ret_32"] = c.pct_change(32)

    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std(ddof=0)
    df["bb_z20"] = (c - ma20) / (std20 + 1e-12)
    df["rv_16"]  = df["ret_1"].rolling(16).std(ddof=0)

    df["slope_log_8"]  = _slope_log(c, 8)
    df["slope_log_16"] = _slope_log(c, 16)
    df["slope_log_32"] = _slope_log(c, 32)

    ema_fast = _ema(c, 12); ema_slow = _ema(c, 48)
    df["ema_fast_dist"]    = (c - ema_fast) / (abs(ema_fast) + 1e-12)
    df["ema_fast_slow_gap"]= (ema_fast - ema_slow) / (abs(ema_slow) + 1e-12)
    df["mom_ratio"]        = (c - ema_fast) / (abs(ema_fast - ema_slow) + 1e-12)

    df["vol_chg"] = v.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0)
    df["vol_z48"] = _zscore(v, 48)

    df["atr14"]     = _atr(df, 14)
    df["atr_ratio"] = df["atr14"] / (c + 1e-12)

    df["rsi14_15m"] = _rsi(c, 14)

    di = df.set_index("timestamp")
    c_1h = di["close"].resample("1h").last().dropna()
    c_4h = di["close"].resample("4h").last().dropna()
    rsi_1h = _rsi(c_1h, 14); rsi_4h = _rsi(c_4h, 14)
    # 對齊回 15m
    aux1 = pd.DataFrame({"timestamp": rsi_1h.index, "_v": rsi_1h.values})
    aux4 = pd.DataFrame({"timestamp": rsi_4h.index, "_v": rsi_4h.values})
    df = pd.merge_asof(df.sort_values("timestamp"), aux1.sort_values("timestamp"),
                       on="timestamp", direction="backward")
    df.rename(columns={"_v":"rsi14_1h"}, inplace=True)
    df = pd.merge_asof(df.sort_values("timestamp"), aux4.sort_values("timestamp"),
                       on="timestamp", direction="backward")
    df.rename(columns={"_v":"rsi14_4h"}, inplace=True)

    # 清理
    df = df.replace([np.inf,-np.inf], np.nan)
    # 只去掉特徵計算不足的前期，保留最後一列
    df = df.dropna(subset=FEATURES, how="any")
    return df.reset_index(drop=True)

# ================== 濾網（與訓練一致） ==================
def build_regime_masks(px: pd.DataFrame):
    c = px["close"]
    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std(ddof=0)
    bb_z = (c - ma20) / (std20 + 1e-12)
    d = c.diff()
    rs = (d.clip(lower=0).rolling(14).mean() /
          ((-d.clip(upper=0)).rolling(14).mean() + 1e-12))
    rsi14 = 100 - (100/(1+rs))
    bb_z = bb_z.fillna(0); rsi14 = rsi14.fillna(50)

    # 訓練時的濾網
    BBZ_ZMIN = 0.25; RSI_LONG_MAX = 35; RSI_SHORT_MIN = 65
    mask_long  = (bb_z >=  BBZ_ZMIN) | (rsi14 <= RSI_LONG_MAX)
    mask_short = (bb_z <= -BBZ_ZMIN) | (rsi14 >= RSI_SHORT_MIN)

    # 低波動/異常首根剔除
    atr_ratio = (px["atr14"] / (px["close"] + 1e-12)).fillna(0)
    ret1_abs  = px["close"].pct_change(1).abs().fillna(0)
    liq_mask  = (atr_ratio >= 0.003) & (ret1_abs <= 0.01)

    mask_long  = (mask_long.values  & liq_mask.values)
    mask_short = (mask_short.values & liq_mask.values)
    return mask_long, mask_short

# ================== 市況與 TP/SL ==================
def build_tp_sl_prices(side: str, entry_price: float, atr: float) -> tuple[float, float]:
    if side == "LONG":
        sl = round(entry_price - SL_L * atr, 2)
        tp = round(entry_price + TP_L * atr, 2)
    else:
        sl = round(entry_price + SL_S * atr, 2)
        tp = round(entry_price - TP_S * atr, 2)
    return sl, tp

def tighten_stop_only(side: str, current_sl: float, entry_price: float, atr_now: float) -> float:
    # 只收緊，不放寬；1.2×ATR 的追蹤
    if side == "LONG":
        proposed = round(entry_price - 1.2 * atr_now, 2)
        return max(current_sl, proposed)
    else:
        proposed = round(entry_price + 1.2 * atr_now, 2)
        return min(current_sl, proposed)

# ================== 推論 ==================
def predict_prob(df15: pd.DataFrame):
    feats = build_features_live(df15)
    if feats.empty:
        return None, None, None, None

    X = feats[FEATURES].iloc[[-1]]            # keep last row as DataFrame
    X_np = X.to_numpy(dtype=float, copy=False) # <<< 轉成 numpy，避免警告
    X_scaled = scaler.transform(X_np)          # <<< 用 numpy 給 scaler

    p_up = float(clf.predict_proba(X_scaled)[0,1])
    p_dn = 1.0 - p_up

    px = feats[["timestamp","open","high","low","close","atr14"]].copy()
    mL, mS = build_regime_masks(px)
    regime_long_ok  = bool(mL[-1])
    regime_short_ok = bool(mS[-1])

    atr_now = float(feats["atr14"].iloc[-1])
    last_row = feats.iloc[-1]
    return p_dn, p_up, atr_now, dict(regime_long_ok=regime_long_ok, regime_short_ok=regime_short_ok), last_row

# ================== 主流程 ==================
def main():
    df = fetch_klines_and_append_to_local()
    dn_prob, up_prob, atr_now, regime, latest = predict_prob(df)

    if dn_prob is None:
        notify(summary=None, signals=None,
               current_price=float(df["close"].iloc[-1]) if not df.empty else None,
               extra_msg="❌ 無法預測：特徵不足")
        return

    current_price = float(df["close"].iloc[-1])
    position = load_current_position(POSITION_PATH)

    # ===== 已持倉：檢查 TP/SL 與超時等 =====
    if position:
        side = position.get("side", "LONG")
        entry_price = float(position["entry_price"])
        bars_held = int(position.get("bars_held", 0)) + 1
        position["bars_held"] = bars_held

        sl = position.get("sl"); tp = position.get("tp")
        hit_tp = hit_sl = False
        if side == "LONG":
            if tp is not None and current_price >= tp: hit_tp = True
            elif sl is not None and current_price <= sl: hit_sl = True
        else:
            if tp is not None and current_price <= tp: hit_tp = True
            elif sl is not None and current_price >= sl: hit_sl = True

        if hit_tp or hit_sl:
            rtn = (entry_price - current_price)/entry_price if side=="SHORT" else (current_price - entry_price)/entry_price
            trade = {
                "entry_time": position["entry_time"],
                "exit_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "entry_price": float(entry_price),
                "exit_price":  float(current_price),
                "return": round(rtn, 6),
                "holding_minutes": int(bars_held * 15),
                "horizon": "cls_h16",
                "side": side,
                "reason": "TP" if hit_tp else "SL",
                "tp_sl_mode": "ATR_FIXED_FROM_OPT",
                "sl": float(position.get("sl", entry_price)),
                "tp": float(position.get("tp", entry_price)),
                "sl_mult": float(position.get("sl_mult", 0.0)),
                "tp_mult": float(position.get("tp_mult", 0.0)),
                "atr_at_entry": float(position.get("atr_at_entry", 0.0)),
                "atr_n": ATR_N,
                "bars_held_close": int(bars_held),
                "max_hold_bars": MAX_HOLD_BARS,
            }
            log_trade(trade); sync_trade_to_sheet(trade)
            notify(summary={"cls":[up_prob]}, current_price=current_price,
                   holding=False, entry_price=entry_price, entry_time=position["entry_time"],
                   side=side, sl=sl, tp=tp,
                   reason=("TP" if hit_tp else "SL"),
                   extra_msg=f"🔔 觸發 {'TP' if hit_tp else 'SL'}：{current_price:.2f}")
            clear_position(POSITION_PATH)
            return

        # 既有 exit（超時等）
        exit_info = check_exit_condition(position, current_price)
        if exit_info["exit"]:
            rtn = (entry_price - current_price)/entry_price if side=="SHORT" else (current_price - entry_price)/entry_price
            trade = {
                "entry_time": position["entry_time"],
                "exit_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "entry_price": float(entry_price),
                "exit_price":  float(current_price),
                "return": round(rtn, 6),
                "holding_minutes": int(exit_info["holding_minutes"]),
                "horizon": "cls_h16",
                "side": side,
                "reason": str(exit_info.get("reason","OTHER")),
                "tp_sl_mode": "ATR_FIXED_FROM_OPT",
                "sl": float(position.get("sl", entry_price)),
                "tp": float(position.get("tp", entry_price)),
                "sl_mult": float(position.get("sl_mult", 0.0)),
                "tp_mult": float(position.get("tp_mult", 0.0)),
                "atr_at_entry": float(position.get("atr_at_entry", 0.0)),
                "atr_n": ATR_N,
                "bars_held_close": int(position.get("bars_held", 0)),
                "max_hold_bars": MAX_HOLD_BARS,
            }
            log_trade(trade); sync_trade_to_sheet(trade)
            notify(summary={"cls":[up_prob]}, current_price=current_price,
                   holding=False, entry_price=entry_price, entry_time=position["entry_time"],
                   side=side, sl=position.get("sl"), tp=position.get("tp"),
                   reason=str(exit_info.get("reason","OTHER")),
                   extra_msg="🔔 觸發非 TP/SL 出場條件")
            clear_position(POSITION_PATH)
            return

        # 追蹤止損（只收緊）
        unreal = (current_price - entry_price)/entry_price if side=="LONG" else (entry_price - current_price)/entry_price
        if unreal > 0.005 or bars_held >= MIN_HOLD_BARS:
            new_sl = tighten_stop_only(side, position.get("sl", entry_price), entry_price, atr_now)
            if new_sl != position.get("sl"):
                position["sl"] = new_sl

        # 回報持倉
        save_current_position(position, POSITION_PATH)
        notify(summary={"cls":[up_prob]}, current_price=current_price,
               holding=True, entry_price=entry_price, entry_time=position["entry_time"],
               side=side, sl=position.get("sl"), tp=position.get("tp"),
               extra_msg=f"方向：{side}\n已持有：{bars_held*15} 分鐘\nTP：{position.get('tp',0):.2f} / SL：{position.get('sl',0):.2f}")
        return

    # ===== 無持倉：進場判斷（雙門檻 + 濾網）=====
    side = None
    long_ok  = (up_prob  >= TH_LONG)  and regime["regime_long_ok"]
    short_ok = (dn_prob  >= TH_SHORT) and regime["regime_short_ok"]
    if long_ok: side = "LONG"
    elif short_ok: side = "SHORT"

    if side:
        entry_price = current_price
        sl, tp = build_tp_sl_prices(side, entry_price, atr_now)
        position = {
            "entry_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": float(entry_price),
            "horizon": "cls_h16",
            "side": side,
            "sl": float(sl),
            "tp": float(tp),
            "atr_at_entry": float(atr_now),
            "atr_n": ATR_N,
            "tp_sl_mode": "ATR_FIXED_FROM_OPT",
            "sl_mult": float(SL_L if side=="LONG" else SL_S),
            "tp_mult": float(TP_L if side=="LONG" else TP_S),
            "bars_held": 0,
            "min_hold_bars": MIN_HOLD_BARS,
            "max_hold_bars": MAX_HOLD_BARS
        }
        save_current_position(position, POSITION_PATH)

        msg = (f"➡️ 進場：{side}\n"
               f"p_up={up_prob:.3f} / p_dn={dn_prob:.3f}\n"
               f"門檻 L={TH_LONG:.3f} / S={TH_SHORT:.3f}\n"
               f"ATR×(SL={position['sl_mult']:.2f}, TP={position['tp_mult']:.2f})\n"
               f"TP={tp:.2f} / SL={sl:.2f}")
        notify(summary={"cls":[up_prob]}, signals=True, current_price=current_price,
               is_regression=False, side=side, sl=sl, tp=tp,
               sl_mult=position["sl_mult"], tp_mult=position["tp_mult"],
               tp_sl_mode="ATR_FIXED_FROM_OPT", atr_now=atr_now, extra_msg=msg)
    else:
        msg = (f"➡️ 無進場\n"
               f"p_up={up_prob:.3f} / p_dn={dn_prob:.3f} | "
               f"門檻 L={TH_LONG:.3f} / S={TH_SHORT:.3f}")
        notify(summary={"cls":[up_prob]}, signals=False, current_price=current_price,
               is_regression=False, extra_msg=msg)

if __name__ == "__main__":
    while True:
        main()
        # 等到下一個 15 分鐘整點
        now = datetime.utcnow()
        sleep_sec = (15 - (now.minute % 15)) * 60 - now.second
        if sleep_sec < 0:
            sleep_sec += 15*60
        print(f"⏳ 等待 {sleep_sec} 秒到下一個15分鐘...")
        time.sleep(sleep_sec)
