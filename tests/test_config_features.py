import pathlib, sys, yaml
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from csp.utils.config import get_symbol_features
from csp.optimize.feature_opt import apply_best_params_to_cfg


def _sample_cfg():
    return {
        "features": {
            "default": {
                "ema_windows": [9, 21, 50],
                "h4_rule": {"enabled": True, "resample": "4H"},
                "rsi": {"enabled": True, "window": 14},
                "bollinger": {"enabled": True, "window": 20, "std": 2.0},
                "atr": {"enabled": True, "window": 14},
            },
            "per_symbol": {
                "BTCUSDT": {
                    "rsi": {"enabled": True, "window": 20},
                    "bollinger": {"enabled": True, "window": 24, "std": 3.42},
                    "atr": {"enabled": True, "window": 17},
                }
            },
        },
    }


def test_get_symbol_features_merge():
    cfg = _sample_cfg()
    btc = get_symbol_features(cfg, "BTCUSDT")
    assert btc["rsi_window"] == 20
    assert btc["bb_window"] == 24
    assert abs(btc["bb_std"] - 3.42) < 1e-9
    assert btc["atr_window"] == 17

    eth = get_symbol_features(cfg, "ETHUSDT")
    assert eth["rsi_window"] == 14
    assert eth["bb_window"] == 20
    assert eth["atr_window"] == 14


def test_apply_best_params(tmp_path):
    cfg = _sample_cfg()
    cfg_path = tmp_path / "s.yaml"
    cfg_path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    best = {"rsi_window": 8, "bb_window": 30, "bb_std": 1.5, "atr_window": 10}
    log_file = tmp_path / "log.txt"
    apply_best_params_to_cfg(cfg_path, "ETHUSDT", best, apply=True, log_file=log_file)
    updated = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    eth_cfg = updated["features"]["per_symbol"]["ETHUSDT"]
    assert eth_cfg["rsi"]["window"] == 8
    assert eth_cfg["bollinger"]["window"] == 30
    assert float(eth_cfg["bollinger"]["std"]) == 1.5
    assert eth_cfg["atr"]["window"] == 10
    assert log_file.exists()
