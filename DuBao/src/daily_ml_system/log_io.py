# -*- coding: utf-8 -*-
"""Lưu / đọc prediction_log.csv"""
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from . import config

COLUMNS = [
    "target_date",
    "prediction_date",
    "rf_pred",
    "xgb_pred",
    "et_pred",
    "rf_prob",
    "xgb_prob",
    "et_prob",
    "actual",
    "error_rf",
    "error_xgb",
    "error_et",
]


def _ensure_log(path: str) -> None:
    if not os.path.exists(path):
        pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)

def normalize_log_schema(path: Optional[str] = None) -> None:
    """
    Chuẩn hoá file prediction_log.csv nếu trước đây có schema cũ (lr_pred/error_lr...).
    Giữ lại dữ liệu cũ, thêm cột mới nếu thiếu.
    """
    path = path or config.PREDICTION_LOG
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COLUMNS]
    df.to_csv(path, index=False)


def append_prediction(
    target_date: Any,
    prediction_date: Any,
    preds: Dict[str, float],
    path: Optional[str] = None,
) -> None:
    """Thêm hoặc cập nhật một dòng dự đoán (chưa có actual)."""
    path = path or config.PREDICTION_LOG
    _ensure_log(path)
    normalize_log_schema(path)
    log = pd.read_csv(path)
    if len(log) and "target_date" in log.columns:
        log["target_date"] = pd.to_datetime(log["target_date"], errors="coerce", format="mixed").dt.normalize()
    td = pd.Timestamp(target_date).normalize()
    row = {
        "target_date": td.strftime("%Y-%m-%d"),
        "prediction_date": pd.Timestamp(prediction_date).strftime("%Y-%m-%d"),
        "rf_pred": preds.get("rf", 0.0),
        "xgb_pred": preds.get("xgb", 0.0),
        "et_pred": preds.get("et", 0.0),
        "rf_prob": preds.get("rf_prob"),
        "xgb_prob": preds.get("xgb_prob"),
        "et_prob": preds.get("et_prob"),
        "actual": None,
        "error_rf": None,
        "error_xgb": None,
        "error_et": None,
    }
    new_row = pd.DataFrame([row], columns=COLUMNS)
    if len(log) and (log["target_date"] == td).any():
        idx = log.index[log["target_date"] == td][0]
        for k, v in row.items():
            if k != "actual":
                log.at[idx, k] = v
    else:
        log = pd.concat([log, new_row], ignore_index=True)
    log.to_csv(path, index=False)


def update_actual(
    target_date: Any,
    actual_mm: float,
    path: Optional[str] = None,
) -> pd.Series:
    """
    Cập nhật lượng mưa thực tế và tính sai số cho từng mô hình.
    Trả về dòng đã sửa.
    """
    path = path or config.PREDICTION_LOG
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chưa có file log: {path}")
    normalize_log_schema(path)
    log = pd.read_csv(path)
    log["target_date"] = pd.to_datetime(log["target_date"], errors="coerce", format="mixed").dt.normalize()
    td = pd.Timestamp(target_date).normalize()
    m = log["target_date"] == td
    if not m.any():
        raise ValueError(f"Không có dòng dự đoán cho target_date={td.date()}")
    idx = log.index[m][0]
    a = float(actual_mm)
    log.at[idx, "actual"] = a
    for col, pred_col in [("error_rf", "rf_pred"), ("error_xgb", "xgb_pred"), ("error_et", "et_pred")]:
        log.at[idx, col] = abs(float(log.at[idx, pred_col]) - a)
    log.to_csv(path, index=False)
    return log.loc[idx]


def read_log(path: Optional[str] = None) -> pd.DataFrame:
    path = path or config.PREDICTION_LOG
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def run_daily_predict_and_log() -> Dict[str, Any]:
    """Chạy predict ngày mai + ghi log (một lệnh cho cron)."""
    from .predict_daily import predict_tomorrow

    target_date, prediction_date, preds, probs = predict_tomorrow()
    # gộp prob vào preds để log
    preds = dict(preds)
    preds["rf_prob"] = probs.get("rf")
    preds["xgb_prob"] = probs.get("xgb")
    preds["et_prob"] = probs.get("et")
    append_prediction(target_date, prediction_date, preds)
    return {
        "target_date": str(target_date.date()),
        "prediction_date": str(prediction_date.date()),
        "preds": preds,
        "log": config.PREDICTION_LOG,
    }
