# -*- coding: utf-8 -*-
"""Dự đoán mưa ngày mai bằng cả 3 mô hình."""
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
) -> Tuple[pd.Timestamp, pd.Timestamp, Dict[str, float]]:
    """
    Trả về (target_date, prediction_date, {'rf':..., 'xgb':..., 'lr':...})
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
    out = {
        "rf": float(models["rf"].predict(Xs)[0]),
        "xgb": float(models["xgb"].predict(Xs)[0]),
        "lr": float(models["lr"].predict(Xs)[0]),
    }
    for k in out:
        out[k] = max(0.0, out[k])
    target_date = prediction_date + timedelta(days=1)
    return target_date, prediction_date, out


def main():
    t, p, preds = predict_tomorrow()
    print(f"prediction_date (feature tới): {p.date()}")
    print(f"target_date (dự đoán mưa): {t.date()}")
    print(preds)


if __name__ == "__main__":
    main()
