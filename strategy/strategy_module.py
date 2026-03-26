
def generate_trade_signal(pred_probs, close_prices=None, threshold_long=0.75, threshold_short=0.25):
    """
    根據預測機率產生交易訊號與止盈/止損：
    - 機率 > threshold_long：做多（+5%/-2%）
    - 機率 < threshold_short：做空（-5%/+2%）
    - 其餘：不交易
    """
    signals = []
    for i, prob in enumerate(pred_probs):
        signal = {
            "signal": "HOLD",
            "take_profit": None,
            "stop_loss": None,
            "reason": f"Prob={prob:.2%}"
        }

        if close_prices is not None:
            close = close_prices[i]
            if prob > threshold_long:
                signal["signal"] = "LONG"
                signal["take_profit"] = close * 1.02
                signal["stop_loss"] = close * 0.99
                signal["reason"] = f"LONG P={prob:.2%}"
            elif prob < threshold_short:
                signal["signal"] = "SHORT"
                signal["take_profit"] = close * 0.98
                signal["stop_loss"] = close * 1.01
                signal["reason"] = f"SHORT P={prob:.2%}"

        signals.append(signal)

    return signals
