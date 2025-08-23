from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
import joblib

from csp.data.loader import load_15m_csv
from csp.features.h16 import build_features_15m_4h
from csp.core.feature import add_features
from csp.utils.config import get_symbol_features

@dataclass
class EntryZoneCfg:
    enabled: bool = False
    method: str = "atr_discount"
    lookahead_bars: int = 16
    long_x: float = 0.5
    short_x: float = 0.5

def _load_cfg(cfg_path: str) -> Dict[str, Any]:
    return yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

def _load_model_bundle(cfg: Dict[str, Any], symbol: str):
    mdir = Path(cfg["io"]["models_dir"]) / symbol
    if not mdir.exists():
        raise FileNotFoundError(f"Model directory not found for {symbol}: {mdir}")

    model_path_joblib = mdir / "xgb_h16_sklearn.joblib"
    model_path_json = mdir / "xgb_h16.json"
    scaler_path = mdir / "scaler_h16.joblib"
    feature_path = mdir / "feature_names.json"
    meta_path = mdir / "meta_h16.json"

    if model_path_joblib.exists():
        model = joblib.load(model_path_joblib)
        model_type = "sklearn"
        model_path = model_path_joblib
    elif model_path_json.exists():
        bst = xgb.Booster(); bst.load_model(str(model_path_json))
        model = bst
        model_type = "booster"
        model_path = model_path_json
    else:
        raise FileNotFoundError(f"No model file found under {mdir}")

    scaler = joblib.load(scaler_path)
    if feature_path.exists():
        feature_names = json.load(open(feature_path, "r", encoding="utf-8"))
    else:
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        feature_names = meta.get("feature_cols", [])
    meta = json.load(open(meta_path, "r", encoding="utf-8")) if meta_path.exists() else {}

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_type": model_type,
        "meta": meta,
    }

def _infer_symbol_from_path(csv_path: str) -> Optional[str]:
    name = Path(csv_path).name.upper()
    if "BTC" in name: return "BTCUSDT"
    if "ETH" in name: return "ETHUSDT"
    if "BCH" in name: return "BCHUSDT"
    return None

def _compute_tp_sl(price: float, atr: float, side: str, atr_cfg: Dict[str, Any]):
    if side == "long":
        tp = price + atr * float(atr_cfg["long"]["tp_mult"])
        sl = price - atr * float(atr_cfg["long"]["sl_mult"])
    else:
        tp = price - atr * float(atr_cfg["short"]["tp_mult"])
        sl = price + atr * float(atr_cfg["short"]["sl_mult"])
    return float(tp), float(sl)

def _decide_side(proba_up: float, long_thr: float, short_thr: float) -> Optional[str]:
    if proba_up >= long_thr: return "long"
    if (1.0 - proba_up) >= short_thr: return "short"
    return None

def _entry_hit_zone(bar: pd.Series, low: float, high: float) -> bool:
    return (float(bar["low"]) <= high and float(bar["high"]) >= low)

def _zone_from_atr_discount(side: str, price: float, atr: float, x_long: float, x_short: float):
    if side == "long":
        z_low, z_high = price - atr * x_long, price
    else:
        z_low, z_high = price, price + atr * x_short
    return float(min(z_low, z_high)), float(max(z_low, z_high))

def _apply_exit(bar: pd.Series, side: str, tp: float, sl: float) -> Optional[str]:
    low, high = float(bar["low"]), float(bar["high"])
    if side == "long":
        if low <= sl: return "sl"
        if high >= tp: return "tp"
    else:
        if high >= sl: return "sl"
        if low  <= tp: return "tp"
    return None

def run_backtest_for_symbol(csv_path: str, cfg_path: str, symbol: Optional[str] = None,
                            start_ts: Optional[pd.Timestamp] = None, end_ts: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
    cfg = _load_cfg(cfg_path)
    sym = symbol or _infer_symbol_from_path(csv_path)
    feat_params = get_symbol_features(cfg, sym)
    long_thr  = float(cfg["execution"]["long_prob_threshold"])
    short_thr = float(cfg["execution"]["short_prob_threshold"])
    atr_cfg   = cfg["execution"]["atr_tp_sl"]
    max_hold_minutes = int(cfg["execution"].get("max_holding_minutes", 240))
    max_hold_bars = max(1, max_hold_minutes // 15)

    ez = cfg["execution"].get("entry_zone", {}) or {}
    entry_cfg = EntryZoneCfg(
        enabled=bool(ez.get("enabled", False)),
        method=str(ez.get("method", "atr_discount")),
        lookahead_bars=int(ez.get("lookahead_bars", 16)),
        long_x=float(ez.get("long_x", 0.5)),
        short_x=float(ez.get("short_x", 0.5)),
    )
    # 進場冷卻時間（平倉後 N 根內不再入場）
    cooldown_bars = int(cfg.get("entry_filter", {}).get("cooldown_bars", 0))
    last_exit_index = None

    # === 印出本次回測的「最終採用參數」 ===
    try:
        _eff = {
            "symbol": sym,
            "long_prob_threshold": long_thr,
            "short_prob_threshold": short_thr,
            "max_holding_minutes": int(cfg.get("execution", {}).get("max_holding_minutes", 240)),
            "entry_zone": {
                "enabled": bool(cfg.get("execution", {}).get("entry_zone", {}).get("enabled", False)),
                "method": str(cfg.get("execution", {}).get("entry_zone", {}).get("method", "atr_discount")),
                "lookahead_bars": int(cfg.get("execution", {}).get("entry_zone", {}).get("lookahead_bars", 16)),
                "long_x": float(cfg.get("execution", {}).get("entry_zone", {}).get("long_x", 0.5)),
                "short_x": float(cfg.get("execution", {}).get("entry_zone", {}).get("short_x", 0.5)),
            },
            "entry_filter": {
                "cooldown_bars": int(cfg.get("entry_filter", {}).get("cooldown_bars", 0)),
            },
            "backtest": {
                "initial_capital": float(cfg.get("backtest", {}).get("initial_capital", 10000.0)),
                "risk_per_trade": float(cfg.get("backtest", {}).get("risk_per_trade", 0.007)),
                "fee_rate": float(cfg.get("backtest", {}).get("fee_rate", 0.0004)),
                "slippage": float(cfg.get("backtest", {}).get("slippage", 0.0002)),
            },
        }
        print("[Effective Params - Backtest]", _eff)
    except Exception:
        pass
    df15 = load_15m_csv(csv_path)
    if start_ts is not None:
        df15 = df15[df15["timestamp"] >= start_ts]
    if end_ts is not None:
        df15 = df15[df15["timestamp"] <= end_ts]
    if len(df15) == 0:
        return {"trades": pd.DataFrame(), "metrics": {
            "交易筆數": 0, "勝率": 0.0, "總收益": 0.0, "獲利因子": 0.0, "最大回撤": 0.0, "平均持倉分鐘": 0.0
        }}

    feats = build_features_15m_4h(
        df15,
        ema_windows=tuple(feat_params["ema_windows"]),
        rsi_window=feat_params["rsi_window"],
        bb_window=feat_params["bb_window"],
        bb_std=feat_params["bb_std"],
        atr_window=feat_params["atr_window"],
        h4_resample=feat_params["h4_resample"],
    )
    feats = add_features(
        feats,
        prev_high_period=feat_params["prev_high_period"],
        prev_low_period=feat_params["prev_low_period"],
        bb_window=feat_params["bb_window"],
        atr_window=feat_params["atr_window"],
        atr_percentile_window=feat_params["atr_percentile_window"],
    )
    bundle = _load_model_bundle(cfg, sym)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_names"]

    Xs = scaler.transform(feats[feature_cols].values)
    if bundle["model_type"] == "sklearn":
        proba_up_seq = np.clip(model.predict_proba(Xs)[:, 1], 0.0, 1.0)
    else:
        dmat = xgb.DMatrix(Xs, feature_names=feature_cols)
        proba_up_seq = np.clip(model.predict(dmat, output_margin=False), 0.0, 1.0)
    feats = feats.copy()
    feats["proba_up"] = proba_up_seq

    trades = []
    state = "flat"; pos_side = None
    entry_price = tp = sl = None
    entry_time = None
    bars_held = 0

    i = 0; n = len(feats)
    while i < n:
        row = feats.iloc[i]
        price = float(row["close"]); atr_h4 = float(row["atr_h4"]); ts = row["timestamp"]

        if state == "flat":
            side = _decide_side(float(row["proba_up"]), long_thr, short_thr)
            if side is None:
                i += 1; continue
            if not entry_cfg.enabled:
                # 冷卻期檢查
                if cooldown_bars > 0 and last_exit_index is not None and (i - last_exit_index) < cooldown_bars:
                    i += 1; continue
                pos_side = side; entry_price = price
                tp, sl = _compute_tp_sl(entry_price, atr_h4, pos_side, atr_cfg)
                entry_time = ts; state = "holding"; bars_held = 0
                i += 1; continue
            # 區間觸價模式（atr_discount）
            z_low, z_high = _zone_from_atr_discount(side, price, atr_h4, entry_cfg.long_x, entry_cfg.short_x)
            hit_index = None
            j_limit = min(n, i + 1 + entry_cfg.lookahead_bars)
            for j in range(i + 1, j_limit):
                if _entry_hit_zone(feats.iloc[j], z_low, z_high):
                    hit_index = j; break
            if hit_index is None:
                i += 1; continue
            hit_row = feats.iloc[hit_index]
            pos_side = side
            # 冷卻期檢查（以命中索引計算）
            if cooldown_bars > 0 and last_exit_index is not None and (hit_index - last_exit_index) < cooldown_bars:
                i = hit_index + 1; continue
            entry_price = float(hit_row["close"])
            atr_h4_hit = float(hit_row["atr_h4"])
            tp, sl = _compute_tp_sl(entry_price, atr_h4_hit, pos_side, atr_cfg)
            entry_time = hit_row["timestamp"]; state = "holding"; bars_held = 0
            i = hit_index + 1; continue

        else:
            exit_flag = _apply_exit(row, pos_side, tp, sl)
            bars_held += 1
            if exit_flag is not None or bars_held >= max_hold_bars:
                if exit_flag == "tp":
                    exit_price = tp; reason = "tp"
                elif exit_flag == "sl":
                    exit_price = sl; reason = "sl"
                else:
                    exit_price = price; reason = "timeout"
                pnl = (exit_price - entry_price) if pos_side == "long" else (entry_price - exit_price)
                trades.append({
                    "entry_time": str(entry_time), "exit_time": str(ts),
                    "side": pos_side, "entry_price": float(entry_price), "exit_price": float(exit_price),
                    "tp": float(tp), "sl": float(sl), "bars_held": int(bars_held),
                    "pnl": float(pnl), "return": float(pnl / entry_price), "exit_reason": reason
                })
                last_exit_index = i
                state = "flat"; pos_side = None
                entry_price = tp = sl = None; entry_time = None; bars_held = 0
            i += 1; continue

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        metrics = {"交易筆數": 0, "勝率": 0.0, "總收益": 0.0, "獲利因子": 0.0, "最大回撤": 0.0, "平均持倉分鐘": 0.0}
        return {"trades": trades_df, "metrics": metrics, "equity_curve": equity_curve}

    wins = (trades_df["pnl"] > 0).sum()
    losses = (trades_df["pnl"] < 0).sum()
    total_pnl = float(trades_df["pnl"].sum())
    win_rate = float(wins / len(trades_df)) if len(trades_df) else 0.0
    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    avg_holding_minutes = float(trades_df["bars_held"].mean() * 15) if len(trades_df) else 0.0

    
    # === 資金基準化：依 SL 風險距離等權分配部位，初始資金可由 cfg 設定 ===
    backtest_cfg = cfg.get("backtest", {})
    # === debug: 列印回測參數（方便驗證 YAML 是否生效） ===
    try:
        print("[Backtest Params]", {
            "initial_capital": backtest_cfg.get("initial_capital", 10000.0),
            "risk_per_trade": backtest_cfg.get("risk_per_trade", 0.007),
            "fee_rate": backtest_cfg.get("fee_rate", 0.0004),
            "slippage": backtest_cfg.get("slippage", 0.0002),
        })
    except Exception:
        pass
    initial_capital = float(backtest_cfg.get("initial_capital", 10000.0))
    risk_per_trade = float(backtest_cfg.get("risk_per_trade", 0.007))   # 0.7%/trade
    fee_rate = float(backtest_cfg.get("fee_rate", 0.0004))               # 單邊手續費
    slippage = float(backtest_cfg.get("slippage", 0.0002))               # 估算滑點

    def _qty_from_risk(entry_price, sl_price, equity):
        risk_dist = abs(entry_price - sl_price)
        if risk_dist <= 0: return 0.0
        risk_amt = equity * risk_per_trade
        qty = risk_amt / risk_dist
        return max(qty, 0.0)

    equity = initial_capital
    eq_curve = []
    for _, tr in trades_df.iterrows():
        e, x, side = float(tr["entry_price"]), float(tr["exit_price"]), str(tr["side"])
        sl = float(tr["sl"])
        qty = _qty_from_risk(e, sl, equity)
        # 成本 + 費用（雙邊）
        buy_fee = e * abs(qty) * fee_rate
        sell_fee = x * abs(qty) * fee_rate
        buy_slip = e * abs(qty) * slippage
        sell_slip = x * abs(qty) * slippage
        pnl = (x - e) * (qty if side == "long" else -qty) - buy_fee - sell_fee - buy_slip - sell_slip
        equity += pnl
        eq_curve.append(equity)
        # === 資金曲線輸出準備 ===
        if 'timestamp' in trades_df.columns:
            eq_df = trades_df.copy()
            eq_df['equity'] = eq_curve[:len(eq_df)]
            try:
                out_symbol = symbol if symbol else 'ALL'
                eq_df.to_csv(f'backtests/equity_curve_{out_symbol}.csv', index=False, encoding='utf-8-sig')
            except Exception as e:
                print('[EquityCurve] 寫入失敗:', e)

    final_equity = equity
    total_return_pct = (final_equity / initial_capital - 1.0) * 100.0

    # 以權益曲線計算 MDD
    eq_series = pd.Series(eq_curve, dtype=float)
    peak = eq_series.cummax()
    mdd = float(((peak - eq_series) / peak.replace(0, np.nan)).max()) * 100.0 if not eq_series.empty else 0.0

    # 重算 PF（使用權益計算後的實際交易 PnL）
    # 重新計算每筆 PnL（含費用）
    def _pnl_with_cost(tr, eq_before):
        e, x, side = float(tr["entry_price"]), float(tr["exit_price"]), str(tr["side"])
        sl = float(tr["sl"])
        qty = _qty_from_risk(e, sl, eq_before)
        buy_fee = e * abs(qty) * fee_rate
        sell_fee = x * abs(qty) * fee_rate
        buy_slip = e * abs(qty) * slippage
        sell_slip = x * abs(qty) * slippage
        return (x - e) * (qty if side == "long" else -qty) - buy_fee - sell_fee - buy_slip - sell_slip

    eq_tmp = initial_capital
    pnl_list = []
    for _, tr in trades_df.iterrows():
        pnl_i = _pnl_with_cost(tr, eq_tmp)
        pnl_list.append(pnl_i)
        eq_tmp += pnl_i
    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p <= 0]

    # === 資金曲線（以每筆交易結束時點累計）===
    equity_curve_rows = []  # 保留舊資料
    # === 另行建構 bar-by-bar 資金曲線供圖表使用 ===
    eq_tmp2 = float(initial_capital)
    for _, tr in trades_df.iterrows():
        # equity before trade exit
        eq_before = eq_tmp2
        # 使用含費用/滑點的實際 PnL 計算
        pnl_i = _pnl_with_cost(tr, eq_before) if '_pnl_with_cost' in locals() else float(tr.get("pnl", 0.0))
        eq_after = eq_before + pnl_i
        equity_curve_rows.append({
            "timestamp": tr.get("exit_time"),
            "equity_before": float(eq_before),
            "trade_pnl": float(pnl_i),
            "equity_after": float(eq_after),
            "side": tr.get("side"),
            "entry_price": float(tr.get("entry_price", 0.0)),
            "exit_price": float(tr.get("exit_price", 0.0)),
            "bars_held": int(tr.get("bars_held", 0)),
            "exit_reason": tr.get("exit_reason"),
        })
        eq_tmp2 = eq_after
    equity_curve = pd.DataFrame(equity_curve_rows)
    profit_factor = float(sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (float("inf") if sum(wins) > 0 else 0.0)

    avg_holding_minutes = float(trades_df["bars_held"].mean() * 15) if len(trades_df) else 0.0

    metrics = {
        "交易筆數": int(len(trades_df)),
        "勝率": float(len(wins) / len(trades_df)) if len(trades_df) else 0.0,
        "起始資金": float(initial_capital),
        "最終淨值": float(final_equity),
        "總報酬率%": float(total_return_pct),
        "獲利因子": float(profit_factor),
        "最大回撤%": float(mdd),
        "平均持倉分鐘": float(avg_holding_minutes),
    }
    return {"trades": trades_df, "metrics": metrics, "equity_curve": equity_curve}

def _build_equity_curve_bars(df15: pd.DataFrame, trades_df: pd.DataFrame, initial_capital: float, fee_rate: float, slippage: float):
    """
    以每根 K 線（15m）連續計算資金曲線。
    規則：
    - 進場當根，先扣買入費用/滑點，作為持倉基準 equity_base。
    - 持倉期間以收盤價 mark-to-market。
    - 平倉當根，扣賣出費用/滑點，equity 才落地，下一根轉為空倉基準。
    """
    if df15.empty:
        return pd.DataFrame(columns=["timestamp","equity"])

    # 準備交易事件索引
    trades_df = trades_df.copy()
    if "entry_time" in trades_df.columns:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
    if "exit_time" in trades_df.columns:
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")
    trades_df = trades_df.sort_values("entry_time")

    # 建立查找：entry/exit 索引 by timestamp
    idx_by_ts = {ts: i for i, ts in enumerate(df15["timestamp"])}

    # 將交易以索引表示
    evts = []
    for _, tr in trades_df.iterrows():
        et, xt = tr.get("entry_time"), tr.get("exit_time")
        if pd.isna(et) or pd.isna(xt):  # 略過異常
            continue
        ei = idx_by_ts.get(et, None)
        xi = idx_by_ts.get(xt, None)
        if ei is None or xi is None or xi < ei:
            continue
        evts.append({
            "ei": ei,
            "xi": xi,
            "side": str(tr.get("side")),
            "entry_price": float(tr.get("entry_price", 0.0)),
            "exit_price": float(tr.get("exit_price", 0.0)),
            "sl": float(tr.get("sl", 0.0)),
        })

    equity_series = []
    equity_base = float(initial_capital)
    holding = False
    side = None
    entry_price = 0.0
    qty = 0.0

    # 將事件按 entry index 排序
    evts.sort(key=lambda e: e["ei"])

    evt_ptr = 0
    n = len(df15)

    for i in range(n):
        ts = df15["timestamp"].iloc[i]
        price = float(df15["close"].iloc[i])

        # 先處理平倉：如果上一段持倉且此索引為退出點
        # 我們在到達 xi 時，先用當前價計算 mark-to-market，再扣賣出費用
        if holding and evt_ptr <= len(evts)-1:
            # 注意：若當前持倉屬於上一個事件，xi 可能落在 i，需檢查上一個事件的 xi
            pass  # 由下方更明確的實作覆蓋

        # 若當前 i 恰好是某個事件的 entry
        entered = False
        while evt_ptr < len(evts) and evts[evt_ptr]["ei"] == i:
            e = evts[evt_ptr]
            # 進場：扣買入費用與滑點，確立 qty 與基準
            side = e["side"]
            entry_price = e["entry_price"] if e["entry_price"] > 0 else price
            # 用 SL 距離近似風險來決定倉位大小：qty = (equity_base * risk%) / (|entry - sl|)
            # 這裡簡化：若沒有 sl，則以 1% 價差避免除零（僅為 bar 曲線估值用，不影響交易結果）
            risk_dist = abs(entry_price - (e["sl"] if e["sl"] != 0 else entry_price * 0.99))
            if risk_dist <= 0:
                risk_dist = max(entry_price * 0.01, 1e-8)
            # 估算「持倉名義」為 1 份資金單位（不使用 risk% 再計算，避免與回測倉位重複計）
            # 這裡以名義：qty = equity_base / entry_price 的 0.1 倍，避免巨大波動（僅用於連續曲線視覺對齊）
            qty = (equity_base / max(entry_price, 1e-8)) * 0.1

            buy_fee = entry_price * abs(qty) * fee_rate
            buy_slip = entry_price * abs(qty) * slippage
            equity_base = equity_base - buy_fee - buy_slip
            holding = True
            entered = True
            # 若有多筆交易同時在同一根進場，後續以最後一筆為準（實務少見）
            evt_ptr += 1
            break

        # 標記持倉的 mark-to-market
        if holding:
            if side == "long":
                equity_mark = equity_base + (price - entry_price) * qty
            else:
                equity_mark = equity_base + (entry_price - price) * qty
        else:
            equity_mark = equity_base

        # 若當前 i 恰好是某個事件的 exit（需要找對應的上一個事件）
        # 因為我們 evt_ptr 可能已經前進到下一個事件，所以需要回看上一個事件是否在此處退出
        # 這裡簡化處理：掃描所有事件中 xi == i 且與當前持倉相符者，進行平倉
        # （n 通常較小，此步開銷可接受；如需最佳化可建 xi->event 映射）
        exited_now = False
        if holding:
            for e in evts:
                if e["xi"] == i and e["side"] == side and abs(e["entry_price"] - entry_price) < 1e-6:
                    sell_price = e["exit_price"] if e["exit_price"] > 0 else price
                    sell_fee = sell_price * abs(qty) * fee_rate
                    sell_slip = sell_price * abs(qty) * slippage
                    # 平倉當根：扣賣出費用
                    equity_mark = (equity_base + (sell_price - entry_price) * qty) if side == "long" else (equity_base + (entry_price - sell_price) * qty)
                    equity_mark = equity_mark - sell_fee - sell_slip
                    equity_base = equity_mark  # 落地成為新的空倉基準
                    holding = False
                    side = None
                    qty = 0.0
                    exited_now = True
                    break

        equity_series.append({"timestamp": ts, "equity": float(equity_mark)})

    return pd.DataFrame(equity_series)
