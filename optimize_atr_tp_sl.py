# optimize_atr_tp_sl.py
import pandas as pd, numpy as np
import argparse, os

def load_df(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
    ts = pick("timestamp","open_time","time","date")
    o = pick("open"); h = pick("high","max","h")
    l = pick("low","min","l"); c = pick("close","price","c")
    if ts is None: raise ValueError("找不到時間欄位（timestamp/ time / date）")
    if c  is None: raise ValueError("找不到收盤價欄位（close/ price / c）")
    for col in [o,h,l,c]: 
        if col is not None: df[col]=pd.to_numeric(df[col],errors="coerce")
    df[ts]=pd.to_datetime(df[ts])
    if o is None: o=c
    if h is None: h=c
    if l is None: l=c
    df=df.sort_values(ts).reset_index(drop=True)
    df=df[[ts,o,h,l,c]].rename(columns={ts:"timestamp",o:"open",h:"high",l:"low",c:"close"})
    return df.dropna().reset_index(drop=True)

def compute_atr(df, n=14):
    tr1=(df["high"]-df["low"]).abs()
    tr2=(df["high"]-df["close"].shift(1)).abs()
    tr3=(df["low"] -df["close"].shift(1)).abs()
    tr=np.nanmax(np.vstack([tr1.values,tr2.values,tr3.values]),axis=0)
    df["ATR"]=pd.Series(tr).rolling(n, min_periods=1).mean()
    return df

def grid_search(df, max_horizon=16, sl_grid=None, tp_grid=None):
    if sl_grid is None: sl_grid=[1.0,1.25,1.5,1.75,2.0]  # SL = X*ATR
    if tp_grid is None: tp_grid=[1.5,2.0,2.5,3.0,3.5]    # TP = Y*ATR
    N=len(df)
    # 預先佈局未來 16 根的 high/low
    high_fw=np.full((N,max_horizon),np.nan)
    low_fw =np.full((N,max_horizon),np.nan)
    for h in range(1,max_horizon+1):
        high_fw[:-h,h-1]=df["high"].shift(-h).values[:-h]
        low_fw[:-h, h-1]=df["low" ].shift(-h).values[:-h]

    rec=[]
    for i in range(N-max_horizon-1):
        entry=df.loc[i,"close"]; atr=df.loc[i,"ATR"]
        if not np.isfinite(atr) or atr<=0: continue
        ph=high_fw[i,:]; pl=low_fw[i,:]
        if not np.isfinite(ph).all() or not np.isfinite(pl).all(): continue
        cum_max=np.maximum.accumulate(ph)
        cum_min=np.minimum.accumulate(pl)
        for X in sl_grid:
            sl_price=entry - X*atr
            sl_hit_idx=np.argmax(cum_min<=sl_price)
            sl_hit=(cum_min<=sl_price).any()
            for Y in tp_grid:
                tp_price=entry + Y*atr
                tp_hit_idx=np.argmax(cum_max>=tp_price)
                tp_hit=(cum_max>=tp_price).any()
                if tp_hit and sl_hit:
                    if tp_hit_idx<sl_hit_idx:
                        hit, bars, ret="TP", tp_hit_idx+1, (tp_price-entry)/entry
                    elif sl_hit_idx<tp_hit_idx:
                        hit, bars, ret="SL", sl_hit_idx+1, (sl_price-entry)/entry
                    else:
                        hit, bars, ret="SL", sl_hit_idx+1, (sl_price-entry)/entry
                elif tp_hit:
                    hit, bars, ret="TP", tp_hit_idx+1, (tp_price-entry)/entry
                elif sl_hit:
                    hit, bars, ret="SL", sl_hit_idx+1, (sl_price-entry)/entry
                else:
                    final=df.loc[i+max_horizon,"close"]
                    hit, bars, ret="TIMEOUT", max_horizon, (final-entry)/entry
                rec.append((X,Y,hit,bars,ret))
    res=pd.DataFrame(rec, columns=["X_SL_ATR","Y_TP_ATR","hit","bars_held","return_pct"])
    summary=(res.groupby(["X_SL_ATR","Y_TP_ATR"])
       .agg(trades=("return_pct","count"),
            win_rate=("return_pct",lambda s: np.mean(s>0)),
            avg_return=("return_pct","mean"),
            avg_bars=("bars_held","mean"),
            tp_rate=("hit",lambda s: np.mean(s=="TP")),
            sl_rate=("hit",lambda s: np.mean(s=="SL")),
            timeout_rate=("hit",lambda s: np.mean(s=="TIMEOUT")))
       .reset_index())
    # 一個穩健排序：先看 avg_return，再看 win_rate、再看 trades
    summary=summary.sort_values(["avg_return","win_rate","trades"], ascending=[False,False,False]).reset_index(drop=True)
    return res, summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("csv_path", help="例如: data/btc_15m_data_360days.csv")
    ap.add_argument("--atr", type=int, default=14)
    ap.add_argument("--horizon", type=int, default=16, help="最大持倉根數，預設 16 根 = 4 小時")
    args=ap.parse_args()

    df=load_df(args.csv_path)
    df=compute_atr(df, n=args.atr)
    res, summary=grid_search(df, max_horizon=args.horizon)

    out_dir=os.path.dirname(os.path.abspath(args.csv_path)) or "."
    p1=os.path.join(out_dir,"atr_tp_sl_all_trades_sample.csv")
    p2=os.path.join(out_dir,"atr_tp_sl_grid_summary.csv")
    p3=os.path.join(out_dir,"atr_tp_sl_top15.csv")

    res.sample(min(20000,len(res)), random_state=42).to_csv(p1, index=False)
    summary.to_csv(p2, index=False)
    summary.head(15).to_csv(p3, index=False)

    best=summary.iloc[0]
    print("\n✅ 最佳 ATR 倍數（以 avg_return 優先）")
    print(f"SL = {best['X_SL_ATR']} × ATR, TP = {best['Y_TP_ATR']} × ATR")
    print(f"trades={int(best['trades'])}, win_rate={best['win_rate']:.3f}, avg_return={best['avg_return']:.5f}, avg_bars={best['avg_bars']:.2f}")
    print("\n已輸出：")
    print(f"- {p2}（全組合彙整）")
    print(f"- {p3}（Top 15）")
    print(f"- {p1}（抽樣交易明細）")

if __name__=="__main__":
    main()
