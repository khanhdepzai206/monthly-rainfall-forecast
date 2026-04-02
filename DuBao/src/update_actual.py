#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cập nhật lượng mưa thực tế và tính sai số cho các mô hình.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

def update_actual_rainfall(actual_date, actual_rainfall):
    """
    Cập nhật actual rainfall và tính error cho ngày đã dự đoán.

    Args:
        actual_date (str): Ngày có format 'YYYY-MM-DD'
        actual_rainfall (float): Lượng mưa thực tế (mm)
    """

    log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'prediction_log.csv')

    if not os.path.exists(log_path):
        print("❌ Không tìm thấy prediction log!")
        return False

    # Đọc log
    log_df = pd.read_csv(log_path)

    # Tìm dòng có date tương ứng
    mask = log_df['date'] == actual_date
    if not mask.any():
        print(f"❌ Không tìm thấy prediction cho ngày {actual_date}")
        return False

    # Cập nhật actual
    log_df.loc[mask, 'actual'] = actual_rainfall

    # Tính error cho từng model
    pred_cols = ['rf_pred', 'xgb_pred', 'lr_pred']
    error_cols = ['error_rf', 'error_xgb', 'error_lr']

    for pred_col, error_col in zip(pred_cols, error_cols):
        if pred_col in log_df.columns and log_df.loc[mask, pred_col].notna().any():
            pred_value = log_df.loc[mask, pred_col].iloc[0]
            if pd.notna(pred_value):
                error = abs(pred_value - actual_rainfall)
                log_df.loc[mask, error_col] = error

    # Lưu lại
    log_df.to_csv(log_path, index=False)

    print(f"✓ Đã cập nhật actual rainfall cho ngày {actual_date}: {actual_rainfall} mm")
    print("Errors:")
    for error_col in error_cols:
        if error_col in log_df.columns and log_df.loc[mask, error_col].notna().any():
            error_val = log_df.loc[mask, error_col].iloc[0]
            print(".4f")

    return True

def get_recent_errors(days=7):
    """Lấy sai số trung bình của 7 ngày gần nhất cho từng model."""

    log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'prediction_log.csv')

    if not os.path.exists(log_path):
        return {}

    log_df = pd.read_csv(log_path)

    # Lọc các dòng có actual
    valid_df = log_df.dropna(subset=['actual']).tail(days)

    if len(valid_df) == 0:
        return {}

    errors = {}
    error_cols = ['error_rf', 'error_xgb', 'error_lr']

    for error_col in error_cols:
        if error_col in valid_df.columns:
            mean_error = valid_df[error_col].mean()
            errors[error_col.replace('error_', '')] = mean_error
            print(f"{error_col}: {mean_error:.4f}")

    return errors

if __name__ == "__main__":
    # Example usage
    if len(os.sys.argv) == 3:
        date = os.sys.argv[1]
        rainfall = float(os.sys.argv[2])
        update_actual_rainfall(date, rainfall)
    else:
        print("Usage: python update_actual.py YYYY-MM-DD rainfall_mm")
        print("Example: python update_actual.py 2024-01-15 5.2")