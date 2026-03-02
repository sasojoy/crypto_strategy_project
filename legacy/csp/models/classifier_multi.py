from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import joblib
import datetime


@dataclass
class MultiThresholdClassifier:
    horizons: List[int]
    thresholds: List[float]
    model_type: str = "xgboost"
    random_state: int = 42

    models: Dict[Tuple[int, float], any] = field(default_factory=dict)
    calibrators: Dict[Tuple[int, float], CalibratedClassifierCV] = field(default_factory=dict)
    feature_columns: List[str] | None = None
    cv_scores: Dict[str, float] = field(default_factory=dict)
    skipped: List[Tuple[int, float]] = field(default_factory=list)
    calibrate_method: str | None = None

    # ------------------------------------------------------------------
    def _build_model(self):
        if self.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(random_state=self.random_state)
        elif self.model_type == "randomforest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    # ------------------------------------------------------------------
    def _time_series_cv(self, model, X: np.ndarray, y: pd.Series, cv_cfg: dict | None):
        scores: List[float] = []
        if not cv_cfg:
            return scores
        n_splits = int(cv_cfg.get("n_splits", 5))
        cv_type = cv_cfg.get("type", "expanding")
        purge = int(cv_cfg.get("purge_bars", 0))
        embargo = int(cv_cfg.get("embargo_bars", 0))
        n = len(X)
        if n_splits < 2 or n <= n_splits:
            return scores
        if cv_type == "expanding":
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for tr_idx, te_idx in tscv.split(X):
                if y.iloc[tr_idx].nunique() < 2 or y.iloc[te_idx].nunique() < 2:
                    continue
                m = clone(model)
                m.fit(X[tr_idx], y.iloc[tr_idx])
                try:
                    proba = m.predict_proba(X[te_idx])[:, 1]
                    scores.append(roc_auc_score(y.iloc[te_idx], proba))
                except Exception:
                    continue
        else:  # purged_kfold (simplified)
            fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
            fold_sizes[: n % n_splits] += 1
            current = 0
            indices = np.arange(n)
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                te_idx = indices[start:stop]
                tr_left = indices[: max(0, start - purge)]
                tr_right = indices[min(n, stop + purge + embargo) :]
                tr_idx = np.concatenate([tr_left, tr_right])
                current = stop
                if len(tr_idx) == 0 or len(te_idx) == 0:
                    continue
                if y.iloc[tr_idx].nunique() < 2 or y.iloc[te_idx].nunique() < 2:
                    continue
                m = clone(model)
                m.fit(X[tr_idx], y.iloc[tr_idx])
                try:
                    proba = m.predict_proba(X[te_idx])[:, 1]
                    scores.append(roc_auc_score(y.iloc[te_idx], proba))
                except Exception:
                    continue
        return scores

    # ------------------------------------------------------------------
    def fit(
        self,
        df_features: pd.DataFrame,
        df_labels: pd.DataFrame,
        cv: dict | None = None,
        calibrate: str | None = "isotonic",
    ):
        self.feature_columns = list(df_features.columns)
        self.calibrate_method = calibrate
        for h in self.horizons:
            for t in self.thresholds:
                col = (h, t)
                if col not in df_labels.columns:
                    self.skipped.append(col)
                    continue
                y = df_labels[col].dropna()
                X = df_features.loc[y.index].values
                if y.nunique() < 2 or len(y) < 20:
                    self.skipped.append(col)
                    continue
                model = self._build_model()
                scores = self._time_series_cv(model, X, y, cv)
                if scores:
                    self.cv_scores[f"{h}_{t}"] = float(np.mean(scores))
                model_full = self._build_model()
                model_full.fit(X, y)
                self.models[col] = model_full
                # calibration
                method = calibrate
                cal_obj = None
                if calibrate:
                    if len(y) < 50 or y.nunique() < 2:
                        if method == "isotonic":
                            method = "platt"
                        else:
                            method = None
                    if method:
                        cv_method = "sigmoid" if method == "platt" else "isotonic"
                        try:
                            cal_obj = CalibratedClassifierCV(
                                model_full, method=cv_method, cv=3
                            )
                            cal_obj.fit(X, y)
                        except Exception:
                            cal_obj = None
                if cal_obj:
                    self.calibrators[col] = cal_obj
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, df_features_latest: pd.DataFrame) -> Dict[Tuple[int, float], float]:
        if self.feature_columns is None:
            raise ValueError("Model not fitted")
        X = df_features_latest[self.feature_columns].values
        out: Dict[Tuple[int, float], float] = {}
        for key, model in self.models.items():
            if key in self.calibrators:
                proba = self.calibrators[key].predict_proba(X)[:, 1]
            else:
                proba = model.predict_proba(X)[:, 1]
            out[key] = float(proba[-1])
        for key in self.skipped:
            if key not in out:
                out[key] = float("nan")
        return out

    # ------------------------------------------------------------------
    def save(self, out_dir: str) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        for (h, t), model in self.models.items():
            joblib.dump(model, path / f"model_{h}_{t}.pkl")
            cal = self.calibrators.get((h, t))
            if cal is not None:
                joblib.dump(cal, path / f"cal_{h}_{t}.pkl")
        meta = {
            "horizons": self.horizons,
            "thresholds": self.thresholds,
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "trained_at": datetime.datetime.utcnow().isoformat(),
            "cv_scores": self.cv_scores,
            "skipped": self.skipped,
            "calibrate": self.calibrate_method,
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, in_dir: str) -> "MultiThresholdClassifier":
        path = Path(in_dir)
        meta = json.load(open(path / "meta.json", "r", encoding="utf-8"))
        obj = cls(
            horizons=meta.get("horizons", []),
            thresholds=meta.get("thresholds", []),
            model_type=meta.get("model_type", "xgboost"),
            random_state=meta.get("random_state", 42),
        )
        obj.feature_columns = meta.get("feature_columns")
        obj.cv_scores = meta.get("cv_scores", {})
        obj.skipped = [tuple(x) for x in meta.get("skipped", [])]
        obj.calibrate_method = meta.get("calibrate")
        for h in obj.horizons:
            for t in obj.thresholds:
                key = (h, t)
                model_path = path / f"model_{h}_{t}.pkl"
                if model_path.exists():
                    obj.models[key] = joblib.load(model_path)
                cal_path = path / f"cal_{h}_{t}.pkl"
                if cal_path.exists():
                    obj.calibrators[key] = joblib.load(cal_path)
        return obj
