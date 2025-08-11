# -*- coding: utf-8 -*-
"""
feature_audit_h16.py
針對 15m K 線、預測 Horizon=16(=4小時) 的特徵健檢工具
輸入：btc_15m_data_360days.csv（需含 timestamp, open, high, low, close, volume）
輸出：audit_h16/ 下列報表
- summary_stats.csv   : train/test 各特徵的統計摘要
- drift_ks.csv        : KS 統計（train vs test）與 p-value
- correlation.csv     : 與 y_up 的 point-biserial 相關、互資訊
- psi.csv             : PSI 指標（train vs test）
"""

import os, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score

# ==== 參數 ====
CSV_PATH = "data/btc_15m_data_360days.csv"  # 改路徑也可以
OUT_DIR = Path("audit_h16"); OUT_DIR.mkdir(parents=True, exist_ok=True)
H = 16                # 4小時 horizon
TEST_LAST_DAYS = 90   # 測試集視窗

# ==== 小工具 ====
def rsi(series, window=14):
    d = series.diff()
    up = d.clip(lower=0).rolling(window).mean()
    dn = (-d.clip(upper=0)).rolling(window).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100/(1+rs))

def atr(df, window=14):
    pc = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - pc).abs(),
        (df["low"]  - pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def make_features(df):
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    # 核心 8 特徵（現行策略使用）
    df["ret_1"]  = c.pct_change(1)
    df["ret_4"]  = c.pct_change(4)
    df["ret_16"] = c.pct_change(16)
    rng = (h - l).replace(0, np.nan)
    df["hl_range"] = (h - l) / (c + 1e-12)
    df["body"] = (c - o) / (rng + 1e-12)
    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std(ddof=0)
    df["bb_z"] = (c - ma20) / (std20 + 1e-12)
    df["rsi14"] = rsi(c, 14)
    df["atr14"] = atr(df, 14)

    # 候選補強（僅做檢視，不一定用於訓練）
    # 上下影線、實體比例
    upper = (h - pd.concat([o, c], axis=1).max(axis=1)) / (rng + 1e-12)
    lower = (pd.concat([o, c], axis=1).min(axis=1) - l) / (rng + 1e-12)
    df["upper_shadow"] = upper
    df["lower_shadow"] = lower
    # 多周期斜率（簡易線性回歸斜率）
    for w in [8, 16, 32, 64]:
        def slope_func(s):
            if s.isna().any(): return np.nan
            x = np.arange(len(s), dtype=float)
            a, b = np.polyfit(x, s.values, 1)
            return a
        df[f"slope_{w}"] = c.rolling(w).apply(slope_func, raw=False)
    # 量能標準化
    df["vol_chg"] = v.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0)

    # 標籤
    future_c = c.shift(-H)
    df["y_up"] = (future_c > c).astype(int)

    # 列表：核心 + 候選
    feats_core = ["ret_1","ret_4","ret_16","hl_range","body","bb_z","rsi14","atr14"]
    feats_more = ["upper_shadow","lower_shadow","slope_8","slope_16","slope_32","slope_64","vol_chg"]
    feats_all = feats_core + feats_more
    return df, feats_core, feats_more, feats_all

def point_biserial_corr(x, y):
    # y 是二元(0/1)；回傳係數與 p-value
    x = pd.Series(x).replace([np.inf,-np.inf], np.nan).dropna()
    y = pd.Series(y).iloc[x.index] if hasattr(y, 'iloc') else pd.Series(y).reindex(x.index)
    try:
        r, p = stats.pointbiserialr(x, y)
    except Exception:
        r, p = np.nan, np.nan
    return r, p

def ks_train_test(train, test):
    # 兩樣本 KS 檢驗
    x = pd.Series(train).replace([np.inf,-np.inf], np.nan).dropna()
    y = pd.Series(test ).replace([np.inf,-np.inf], np.nan).dropna()
    if len(x)==0 or len(y)==0: return np.nan, np.nan
    try:
        stat, p = stats.ks_2samp(x, y, alternative="two-sided", method="auto")
    except Exception:
        stat, p = np.nan, np.nan
    return stat, p

def compute_psi(train, test, bins=10):
    # Population Stability Index
    x = pd.Series(train).replace([np.inf,-np.inf], np.nan).dropna()
    y = pd.Series(test ).replace([np.inf,-np.inf], np.nan).dropna()
    if len(x)==0 or len(y)==0: return np.nan
    qs = np.quantile(x, np.linspace(0,1,bins+1))
    qs = np.unique(qs)
    if len(qs) < 2: return np.nan
    x_bins = np.clip(np.digitize(x, qs[1:-1], right=True), 0, len(qs)-2)
    y_bins = np.clip(np.digitize(y, qs[1:-1], right=True), 0, len(qs)-2)
    x_pct = np.bincount(x_bins, minlength=len(qs)-1) / len(x)
    y_pct = np.bincount(y_bins, minlength=len(qs)-1) / len(y)
    eps = 1e-12
    psi = np.sum((x_pct - y_pct) * np.log((x_pct + eps) / (y_pct + eps)))
    return float(psi)

def mutual_info_continuous(x, y, bins=20):
    # 將連續 x 離散化後與 y 計算互資訊
    x = pd.Series(x).replace([np.inf,-np.inf], np.nan).dropna()
    y = pd.Series(y).iloc[x.index] if hasattr(y, 'iloc') else pd.Series(y).reindex(x.index)
    try:
        qs = np.quantile(x, np.linspace(0,1,bins+1))
        qs = np.unique(qs)
        if len(qs) < 2: return np.nan
        x_disc = np.clip(np.digitize(x, qs[1:-1], right=True), 0, len(qs)-2)
        return float(mutual_info_score(x_disc, y.loc[x.index]))
    except Exception:
        return np.nan

# ==== 主流程 ====
def main():
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df, feats_core, feats_more, feats_all = make_features(df)
    df = df.dropna().reset_index(drop=True)

    # Train/Test split
    cutoff = df["timestamp"].max() - pd.Timedelta(days=TEST_LAST_DAYS)
    train_df = df[df["timestamp"] <= cutoff].copy()
    test_df  = df[df["timestamp"] >  cutoff].copy()

    # ========== 1) Summary Stats ==========
    rows = []
    for split_name, d in [("train", train_df), ("test", test_df)]:
        desc = d[feats_all].describe(percentiles=[0.25,0.5,0.75]).T.reset_index().rename(columns={"index":"feature"})
        desc["split"] = split_name
        # 加上 NaN 比例、常數比例
        nan_ratio = d[feats_all].isna().mean()
        nunique = d[feats_all].nunique()
        const_ratio = (nunique <= 1).astype(float)
        desc["nan_ratio"] = desc["feature"].map(nan_ratio.to_dict())
        desc["const_flag"] = desc["feature"].map(const_ratio.to_dict())
        rows.append(desc)
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(OUT_DIR/"summary_stats.csv", index=False, encoding="utf-8-sig")

    # ========== 2) Drift: KS test ==========
    ks_rows = []
    for f in feats_all:
        stat, p = ks_train_test(train_df[f], test_df[f])
        ks_rows.append({"feature": f, "ks_stat": stat, "p_value": p})
    ks_df = pd.DataFrame(ks_rows).sort_values("ks_stat", ascending=False)
    ks_df.to_csv(OUT_DIR/"drift_ks.csv", index=False, encoding="utf-8-sig")

    # ========== 3) Correlation & Mutual Information with y_up ==========
    corr_rows = []
    for f in feats_all:
        r, p = point_biserial_corr(train_df[f], train_df["y_up"])
        mi = mutual_info_continuous(train_df[f], train_df["y_up"], bins=20)
        corr_rows.append({"feature": f, "pb_corr": r, "pb_p_value": p, "mutual_info": mi})
    corr_df = pd.DataFrame(corr_rows).sort_values(["mutual_info","pb_corr"], ascending=[False, False])
    corr_df.to_csv(OUT_DIR/"correlation.csv", index=False, encoding="utf-8-sig")

    # ========== 4) PSI ==========
    psi_rows = []
    for f in feats_all:
        psi = compute_psi(train_df[f], test_df[f], bins=10)
        psi_rows.append({"feature": f, "psi": psi})
    psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)
    psi_df.to_csv(OUT_DIR/"psi.csv", index=False, encoding="utf-8-sig")

    print("✅ 完成！輸出報表位於：", OUT_DIR.resolve())
    print(" - summary_stats.csv")
    print(" - drift_ks.csv")
    print(" - correlation.csv")
    print(" - psi.csv")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
