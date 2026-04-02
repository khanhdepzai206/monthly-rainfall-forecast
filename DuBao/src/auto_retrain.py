#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tự động kiểm tra sai số và retrain models nếu cần.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
from update_actual import get_recent_errors
from train_daily_models import train_daily_models

def check_and_retrain(threshold=2.0, days=7):
    """
    Kiểm tra sai số 7 ngày gần nhất.
    Nếu > threshold thì retrain model đó.

    Args:
        threshold (float): Ngưỡng sai số để retrain
        days (int): Số ngày để tính trung bình
    """

    print(f"🔍 Kiểm tra sai số {days} ngày gần nhất (threshold: {threshold})...")

    errors = get_recent_errors(days)

    if not errors:
        print("⚠ Không có đủ dữ liệu để kiểm tra!")
        return

    models_to_retrain = []

    for model_name, mean_error in errors.items():
        if mean_error > threshold:
            models_to_retrain.append(model_name)
            print(f"⚠ Model {model_name.upper()} cần retrain (error: {mean_error:.4f} > {threshold})")
        else:
            print(f"✓ Model {model_name.upper()} OK (error: {mean_error:.4f})")

    if not models_to_retrain:
        print("🎉 Tất cả models đều hoạt động tốt!")
        return

    # Retrain models cần thiết
    print(f"\n🔄 Retrain {len(models_to_retrain)} model(s)...")

    # Load dữ liệu mới nhất
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    df = pd.read_csv(os.path.join(data_dir, 'daily_features.csv'))

    # Thêm dữ liệu mới từ log (nếu có actual mới)
    log_path = os.path.join(data_dir, 'prediction_log.csv')
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        # Có thể merge actual vào daily_features nếu cần
        # (đơn giản hóa: chỉ retrain với dữ liệu hiện có)

    # Retrain từng model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    import xgboost as xgb
    import pickle

    feature_cols = [c for c in df.columns if 'lag' in c]
    X = df[feature_cols]
    y = df['target']

    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    for model_name in models_to_retrain:
        print(f"\n--- Retraining {model_name.upper()} ---")

        if model_name == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == 'xgb':
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif model_name == 'lr':
            model = LinearRegression()
        else:
            continue

        # Train với toàn bộ dữ liệu
        model.fit(X, y)

        # Lưu model mới
        model_path = os.path.join(models_dir, f'{model_name}_daily_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"✓ Đã retrain và lưu {model_name.upper()}: {model_path}")

    print("🎯 Hoàn thành auto retrain!")

if __name__ == "__main__":
    check_and_retrain()