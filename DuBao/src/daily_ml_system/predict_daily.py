# -*- coding: utf-8 -*-
"""Dự đoán mưa ngày mai bằng 3 mô hình 2-stage (xgb/et/rf)."""
import os
import pickle
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple, TypedDict

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


class TomorrowDetail(TypedDict):
    target_date: pd.Timestamp
    prediction_date: pd.Timestamp
    prob: Dict[str, float]          # P(mưa) theo model (0..1)
    mm_if_rain: Dict[str, float]    # mm nếu mưa (không cắt theo threshold)
    expected_mm: Dict[str, float]   # kỳ vọng mm = prob * mm_if_rain
    thresholds: Dict[str, float]    # threshold theo model (0..1)


def predict_tomorrow_detail(
    data_path: Optional[str] = None,
    bundle_path: Optional[str] = None,
) -> TomorrowDetail:
    """
    Trả về detail dự báo chuyên nghiệp:
    - prob: xác suất mưa
    - mm_if_rain: lượng mưa nếu mưa (regressor)
    - expected_mm: kỳ vọng lượng mưa = prob * mm_if_rain
    """
    path = data_path or config.DAILY_COMBINED
    df = load_daily_combined(path)
    df = merge_actuals_from_log(df, config.PREDICTION_LOG)
    df, feature_cols, prediction_date = build_for_prediction(df)
    bundle = load_bundle(bundle_path)
    models = bundle["models"]
    scaler = bundle["scaler"]

    missing = set(bundle["feature_cols"]) - set(feature_cols)
    if missing:
        raise ValueError(f"Thiếu feature so với lúc train: {missing}")

    X = last_row_feature_matrix(df, bundle["feature_cols"])
    Xs = scaler.transform(X)
    two_stage = bool(bundle.get("two_stage", False))

    rain_th_global = float(bundle.get("rain_prob_threshold", 0.5))
    rain_th_by_model = bundle.get("rain_prob_thresholds") or {}

    prob: Dict[str, float] = {}
    mm_if_rain: Dict[str, float] = {}
    expected_mm: Dict[str, float] = {}
    thresholds: Dict[str, float] = {}

    if two_stage:
        for name in ["xgb", "et", "rf"]:
            pack = models[name]
            clf = pack["clf"]
            reg = pack["reg"]

            if hasattr(clf, "predict_proba"):
                p = float(clf.predict_proba(Xs)[0][1])
            else:
                p = float(clf.predict(Xs)[0])

            pred_log = float(reg.predict(Xs)[0])
            mm = float(np.expm1(pred_log))
            mm = max(0.0, mm)

            th = float(rain_th_by_model.get(name, rain_th_global))

            prob[name] = p
            mm_if_rain[name] = mm
            expected_mm[name] = max(0.0, p * mm)
            thresholds[name] = th
    else:
        # legacy path: coi như expected = pred, prob=0, mm_if_rain=pred
        for k in models:
            v = max(0.0, float(models[k].predict(Xs)[0]))
            prob[k] = 0.0
            mm_if_rain[k] = v
            expected_mm[k] = v
            thresholds[k] = rain_th_global

    target_date = prediction_date + timedelta(days=1)
    return {
        "target_date": target_date,
        "prediction_date": prediction_date,
        "prob": prob,
        "mm_if_rain": mm_if_rain,
        "expected_mm": expected_mm,
        "thresholds": thresholds,
    }


def predict_tomorrow(
    data_path: Optional[str] = None,
    bundle_path: Optional[str] = None,
) -> Tuple[pd.Timestamp, pd.Timestamp, Dict[str, float], Dict[str, float]]:
    """
    Trả về (target_date, prediction_date, preds_mm, probs_rain)
    target_date = ngày cần dự đoán mưa (ngày mai so với ngày cuối trong CSV).
    prediction_date = ngày cuối cùng có đủ feature (hôm nay trong dữ liệu).
    """
    # Legacy API: trả về "mm sau khi cắt theo threshold" để tương thích code cũ.
    d = predict_tomorrow_detail(data_path=data_path, bundle_path=bundle_path)
    preds: Dict[str, float] = {}
    probs: Dict[str, float] = dict(d["prob"])
    for name, p in probs.items():
        th = float(d["thresholds"].get(name, 0.5))
        mm = float(d["mm_if_rain"].get(name, 0.0))
        preds[name] = mm if p >= th else 0.0
    return d["target_date"], d["prediction_date"], preds, probs


def main():
    t, p, preds, probs = predict_tomorrow()
    print(f"prediction_date (feature tới): {p.date()}")
    print(f"target_date (dự đoán mưa): {t.date()}")
    print(preds)
    print("rain_probability:", probs)


if __name__ == "__main__":
    main()
