# -*- coding: utf-8 -*-
"""
train_h16_dynamic_full.py
4小時(=16x15m) Horizon 的分類模型 + 動態ATR止盈止損回測 + 盤勢濾網 +
「日均≈1」目標式門檻搜尋（雙邊二分 + 微調 + 風險懲罰 + 多空不對稱TP/SL）
輸出：
- h16_dynamic/cls_model_h16.pkl
- h16_dynamic/scaler_h16.joblib
- h16_dynamic/opt_h16_dynamic.json
- h16_dynamic/trades_h16.csv
"""
import os, json, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

from model.feature_engineering_h16 import build_features_h16, build_regime_masks_h16

# ===== 設定 =====
OUT_DIR = Path("h16_dynamic"); OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_SOURCE = "auto"                       # "local" | "binance" | "auto"
CSV_PATH = "data/btc_15m_data_360days.csv" # local 檔案位置（auto 會先找這個）
H = 16
TEST_LAST_DAYS = 90
DAILY_TARGET_TPD = 1.0
ALLOW_SHORT = True

# 濾網
BBZ_ZMIN = 0.25
RSI_LONG_MAX = 35
RSI_SHORT_MIN = 65

# 門檻搜尋
BINARY_STEPS = 22
REFINE_STEPS = 9
TH_MIN, TH_MAX = 0.50, 0.95

# 風險目標
DD_CAP = 1200.0
LAMBDA = 0.7  # 懲罰係數

# 非對稱 TP/SL 小網格（可自行加大）
TP_LONG_SET  = [1.6, 1.8]
SL_LONG_SET  = [1.0, 1.2]
TP_SHORT_SET = [1.2, 1.4]
SL_SHORT_SET = [0.8, 1.0]

# ===== 載入資料 =====
def maybe_fetch_from_binance(symbol="BTCUSDT", interval="15m", days=360):
    import requests
    url = "https://api.binance.com/api/v3/klines"
    end = int(time.time()*1000)
    start = end - days*24*60*60*1000
    all_rows = []
    while True:
        params = dict(symbol=symbol, interval=interval, startTime=start, endTime=end, limit=1000)
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows: break
        all_rows.extend(rows)
        last_open_time = rows[-1][0]
        start = last_open_time + 1
        if len(rows) < 1000: break
        time.sleep(0.25)
    cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["timestamp","open","high","low","close","volume"]].dropna().sort_values("timestamp")
    return df.reset_index(drop=True)

def load_data():
    src = DATA_SOURCE
    if src == "auto":
        src = "local" if Path(CSV_PATH).exists() else "binance"
    if src == "local":
        df = pd.read_csv(CSV_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df.sort_values("timestamp").reset_index(drop=True), "local"
    else:
        df = maybe_fetch_from_binance("BTCUSDT","15m",360)
        return df, "binance"

# ===== 回測/目標 =====
def backtest_with_filters(
    proba_up, px, th_long, th_short,
    tpL, slL, tpS, slS,
    mask_long, mask_short, H=16, allow_short=True
):
    pos=0; i=0; n=len(px); trades=[]
    while i < n-1:
        if pos==0:
            go_long  = (proba_up[i] >= th_long)  and mask_long[i]
            go_short = allow_short and ((1.0 - proba_up[i]) >= th_short) and mask_short[i]
            if go_long or go_short:
                side = 1 if go_long else -1
                entry_idx = i+1
                if entry_idx >= n: break
                entry = float(px.loc[entry_idx,"open"])
                atr   = float(px.loc[entry_idx,"atr14"])
                tp = entry + ( (tpL if side==1 else -tpS) * atr )
                sl = entry - ( (slL if side==1 else -slS) * atr )
                j = entry_idx+1; hold=0; exit_price=None; reason=None
                while j < n and hold < H:
                    hi=float(px.loc[j,"high"]); lo=float(px.loc[j,"low"]); cl=float(px.loc[j,"close"])
                    if side==1:
                        if hi>=tp: exit_price,reason = tp,"tp"; break
                        if lo<=sl: exit_price,reason = sl,"sl"; break
                    else:
                        if lo<=tp: exit_price,reason = tp,"tp"; break
                        if hi>=sl: exit_price,reason = sl,"sl"; break
                    hold += 1; j += 1
                if exit_price is None:
                    j = min(j, n-1)
                    exit_price = float(px.loc[j-1,"close"]) if j-1>=0 else float(px.loc[n-1,"close"])
                    reason = "time"
                pnl = (exit_price - entry) * (1 if side==1 else -1)
                trades.append({
                    "side": "long" if side==1 else "short",
                    "entry_idx": int(entry_idx), "exit_idx": int(j),
                    "entry": float(entry), "exit": float(exit_price),
                    "pnl": float(pnl), "hold_bars": int(min(hold+1, H)),
                    "reason": reason,
                    "timestamp_entry": str(px.loc[entry_idx,"timestamp"]),
                    "timestamp_exit" : str(px.loc[min(j,n-1),"timestamp"]),
                })
                i = j
            else:
                i += 1
        else:
            i += 1

    total = len(trades)
    if total==0:
        return {"trades":0,"tpd":0.0,"win_rate":0.0,"total_pnl":0.0,"max_drawdown":0.0,"profit_factor":0.0}, trades

    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    wins_amt = pnl[pnl>0].sum(); losses_amt = -pnl[pnl<0].sum()
    pf = (wins_amt / max(losses_amt,1e-12)) if losses_amt>0 else float("inf")
    wins_n = (pnl>0).sum()
    tpd = total / max(((px["timestamp"].iloc[-1]-px["timestamp"].iloc[0]).total_seconds()/86400.0),1e-9)
    equity = pnl.cumsum(); peak = np.maximum.accumulate(equity); dd = equity - peak
    return {
        "trades": total,
        "tpd": float(tpd),
        "win_rate": float(wins_n/total),
        "total_pnl": float(pnl.sum()),
        "max_drawdown": float(dd.min()),
        "profit_factor": float(pf),
    }, trades

def entries_tpd(proba_up, px, th_long, th_short, mask_long, mask_short):
    long_idx  = np.where((proba_up >= th_long) & mask_long)[0]
    short_idx = np.where(((1.0 - proba_up) >= th_short) & mask_short)[0]
    count = len(np.unique(np.concatenate([long_idx, short_idx])))
    days  = (px["timestamp"].iloc[-1] - px["timestamp"].iloc[0]).total_seconds()/86400.0
    return count / max(days, 1e-9)

def bisect_threshold_for_side(target_tpd, side, other_th, proba_up, px, mask_long, mask_short,
                              steps=22, th_min=0.50, th_max=0.95):
    lo, hi = th_min, th_max
    best = (None, 1e9)
    for _ in range(steps):
        mid = (lo + hi) / 2
        if side == "long":
            tpd = entries_tpd(proba_up, px, mid, other_th, mask_long, mask_short)
        else:
            tpd = entries_tpd(proba_up, px, other_th, mid, mask_long, mask_short)
        diff = abs(tpd - target_tpd)
        if diff < best[1]: best = (mid, diff)
        if tpd > target_tpd: lo = mid
        else: hi = mid
    return float(np.clip(best[0], th_min, th_max))

def objective(res):
    penalty = max(0.0, abs(res["max_drawdown"]) - DD_CAP)
    return (res["total_pnl"] - LAMBDA * penalty, res["profit_factor"], res["win_rate"])

# ===== 主流程 =====
def main():
    t0 = time.time()

    # 1) 讀資料
    df_raw, src = load_data()

    # 2) 特徵
    df_feat, FEATS = build_features_h16(df_raw, horizon_bars=H)
    df_feat = df_feat.dropna().reset_index(drop=True)

    # 3) 切分
    cutoff = df_feat["timestamp"].max() - pd.Timedelta(days=TEST_LAST_DAYS)
    train_df = df_feat[df_feat["timestamp"] <= cutoff].copy()
    test_df  = df_feat[df_feat["timestamp"] >  cutoff].copy()

    X_train, y_train = train_df[FEATS].values, train_df["y_up"].values
    X_test  = test_df[FEATS].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 4) 模型（SGD-Logistic）
    clf = SGDClassifier(loss="log_loss", max_iter=600, random_state=42)
    clf.fit(X_train_s, y_train)
    proba_up = clf.predict_proba(X_test_s)[:,1]

    # 5) 回測資料與濾網
    px = test_df[["timestamp","open","high","low","close","atr14"]].reset_index(drop=True)
    mask_long, mask_short = build_regime_masks_h16(px, BBZ_ZMIN, RSI_LONG_MAX, RSI_SHORT_MIN)

    # 6) 先用二分法各自鎖住 ~0.5 筆/日
    th_long  = bisect_threshold_for_side(DAILY_TARGET_TPD/2, "long",  TH_MAX, proba_up, px, mask_long, mask_short,
                                         steps=BINARY_STEPS, th_min=TH_MIN, th_max=TH_MAX)
    th_short = bisect_threshold_for_side(DAILY_TARGET_TPD/2, "short", TH_MAX, proba_up, px, mask_long, mask_short,
                                         steps=BINARY_STEPS, th_min=TH_MIN, th_max=TH_MAX)

    # 7) 以二分結果為中心 ±0.07 微調
    grid_long  = np.clip(np.linspace(th_long  - 0.07, th_long  + 0.07, REFINE_STEPS), TH_MIN, TH_MAX)
    grid_short = np.clip(np.linspace(th_short - 0.07, th_short + 0.07, REFINE_STEPS), TH_MIN, TH_MAX)

    best=None; best_trades=None
    for tl in grid_long:
        for ts in grid_short:
            tpd_est = entries_tpd(proba_up, px, tl, ts, mask_long, mask_short)
            if not (0.8 <= tpd_est <= 1.2): 
                continue
            for tpL in TP_LONG_SET:
                for slL in SL_LONG_SET:
                    for tpS in TP_SHORT_SET:
                        for slS in SL_SHORT_SET:
                            res, trades = backtest_with_filters(
                                proba_up, px, tl, ts, tpL, slL, tpS, slS,
                                mask_long, mask_short, H=H, allow_short=ALLOW_SHORT
                            )
                            if not (0.8 <= res["tpd"] <= 1.2): 
                                continue
                            if (best is None) or (objective(res) > objective(best)):
                                best, best_trades = {**res, "th_long":float(tl), "th_short":float(ts),
                                                     "tpL":tpL,"slL":slL,"tpS":tpS,"slS":slS}, trades

    # 若微調失敗，至少給二分解的回測
    if best is None:
        # 預設 TP/SL
        res, trades = backtest_with_filters(
            proba_up, px, th_long, th_short, 
            TP_LONG_SET[0], SL_LONG_SET[0], TP_SHORT_SET[0], SL_SHORT_SET[0],
            mask_long, mask_short, H=H, allow_short=ALLOW_SHORT
        )
        best, best_trades = {**res, "th_long":float(th_long), "th_short":float(th_short),
                             "tpL":TP_LONG_SET[0], "slL":SL_LONG_SET[0], "tpS":TP_SHORT_SET[0], "slS":SL_SHORT_SET[0]}, trades

    # 8) 輸出
    with open(OUT_DIR/"cls_model_h16.pkl","wb") as f: pickle.dump(clf, f)
    import joblib; joblib.dump(scaler, OUT_DIR/"scaler_h16.joblib")

    trades_df = pd.DataFrame(best_trades)
    trades_df.to_csv(OUT_DIR/"trades_h16.csv", index=False, encoding="utf-8-sig")

    report = {
        "data_source": src,
        "horizon_bars": H,
        "test_last_days": TEST_LAST_DAYS,
        "daily_target_tpd": DAILY_TARGET_TPD,
        "allow_short": ALLOW_SHORT,
        "filters": {"bbz_zmin": BBZ_ZMIN, "rsi_long_max": RSI_LONG_MAX, "rsi_short_min": RSI_SHORT_MIN},
        "threshold_bounds": [TH_MIN, TH_MAX],
        "binary_steps": BINARY_STEPS,
        "refine_steps": REFINE_STEPS,
        "tp_long_set": TP_LONG_SET,
        "sl_long_set": SL_LONG_SET,
        "tp_short_set": TP_SHORT_SET,
        "sl_short_set": SL_SHORT_SET,
        "features": FEATS,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "best": best,
        "runtime_sec": round(time.time()-t0, 2)
    }
    with open(OUT_DIR/"opt_h16_dynamic.json","w",encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== 完成 ===")
    print("最佳解：", json.dumps(best, ensure_ascii=False))
    print("輸出：", OUT_DIR.resolve())
    print("  - 模型：cls_model_h16.pkl, scaler_h16.joblib")
    print("  - 報告：opt_h16_dynamic.json")
    print("  - 交易：trades_h16.csv")

if __name__ == "__main__":
    main()
