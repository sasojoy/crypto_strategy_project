
import pickle
import numpy as np
import os

# 模型與 scaler 路徑
MODEL_PATH = os.path.join("models", "xgb_cls_model.pkl")
SCALER_PATH = os.path.join("models", "xgb_cls_scaler.pkl")

# 載入模型與標準化器
import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model("models/xgb_cls_model.json")

import joblib
scaler = joblib.load("models/xgb_cls_scaler.joblib")

def predict_up_prob(X_raw_df):
    X_scaled = scaler.transform(X_raw_df)
    prob = model.predict_proba(X_scaled)[:, 1]
    return prob
