# -*- coding: utf-8 -*-
"""Dự đoán mưa ngày mai bằng 3 mô hình 2-stage (xgb/et/rf)."""
import os
import pickle
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import config
from .features import build_for_prediction, last_row_feature_matrix, load_daily_combined, merge_actuals_from_log


def load_bundle(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or config.MODEL_BUNDLE_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"Chưa có mô hình. Chạy: python -m daily_ml_system.cli train — {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def predict_tomorrow(
    data_path: Optional[str] = None,
    bundle_path: Optional[str] = None,
) -> Tuple[pd.Timestamp, pd.Timestamp, Dict[str, float], Dict[str, float]]:
    """
    Trả về (target_date, prediction_date, preds_mm, probs_rain)
    target_date = ngày cần dự đoán mưa (ngày mai so với ngày cuối trong CSV).
    prediction_date = ngày cuối cùng có đủ feature (hôm nay trong dữ liệu).
    """
    path = data_path or config.DAILY_COMBINED
    df = load_daily_combined(path)
    df = merge_actuals_from_log(df, config.PREDICTION_LOG)
    df, feature_cols, prediction_date = build_for_prediction(df)
    bundle = load_bundle(bundle_path)
    models = bundle["models"]
    scaler = bundle["scaler"]
    # đảm bảo cột khớp
    missing = set(bundle["feature_cols"]) - set(feature_cols)
    if missing:
        raise ValueError(f"Thiếu feature so với lúc train: {missing}")
    X = last_row_feature_matrix(df, bundle["feature_cols"])
    Xs = scaler.transform(X)
    two_stage = bool(bundle.get("two_stage", False))
    rain_th = float(bundle.get("rain_prob_threshold", 0.5))

    preds: Dict[str, float] = {}
    probs: Dict[str, float] = {}

    if two_stage:
        for name in ["xgb", "et", "rf"]:
            pack = models[name]
            clf = pack["clf"]
            reg = pack["reg"]
            if hasattr(clf, "predict_proba"):
                prob = float(clf.predict_proba(Xs)[0][1])
            else:
                prob = float(clf.predict(Xs)[0])
            pred_log = float(reg.predict(Xs)[0])
            mm = float(np.expm1(pred_log))
            mm = max(0.0, mm)
            preds[name] = mm if prob >= rain_th else 0.0
            probs[name] = prob
    else:
        # legacy path (old bundle)
        for k in models:
            preds[k] = max(0.0, float(models[k].predict(Xs)[0]))
            probs[k] = 0.0

    target_date = prediction_date + timedelta(days=1)
    return target_date, prediction_date, preds, probs


def main():
    t, p, preds, probs = predict_tomorrow()
    print(f"prediction_date (feature tới): {p.date()}")
    print(f"target_date (dự đoán mưa): {t.date()}")
    print(preds)
    print("rain_probability:", probs)


if __name__ == "__main__":
    main()
