#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dự đoán lượng mưa cho ngày mai bằng 3 mô hình.
Lưu kết quả vào prediction_log.csv
"""
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

def load_models():
    """Load 3 trained models."""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    models = {}

    for name in ['rf', 'xgb', 'lr']:
        model_path = os.path.join(models_dir, f'{name}_daily_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"✓ Loaded {name.upper()} model")
        else:
            print(f"⚠ Model {name.upper()} not found: {model_path}")

    return models

def get_latest_features():
    """Lấy features từ ngày gần nhất để dự đoán ngày mai."""

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    df = pd.read_csv(os.path.join(data_dir, 'daily_features.csv'))

    # Lấy dòng cuối cùng (ngày gần nhất có đủ features)
    latest = df.iloc[-1:].copy()

    # Features cho prediction (giống như trong training)
    exclude = {'date', 'target', 'datetime', 'rainfall'}
    feature_cols = [c for c in df.columns if c not in exclude]
    X_pred = latest[feature_cols]

    # Ngày dự đoán (ngày mai)
    pred_date = pd.to_datetime(latest['date'].iloc[0]) + timedelta(days=1)

    print(f"📊 Using {len(feature_cols)} features for prediction")
    return X_pred, pred_date.date()

def predict_daily():
    """Dự đoán lượng mưa ngày mai bằng cả 3 mô hình."""

    models = load_models()
    if not models:
        print("❌ Không có model nào được load!")
        return None

    # Lấy features
    X_pred, pred_date = get_latest_features()
    print(f"📅 Dự đoán cho ngày: {pred_date}")

    predictions = {}

    # Predict với từng model
    for name, model in models.items():
        try:
            pred = model.predict(X_pred)[0]
            predictions[name] = max(0, pred)  # Không âm
            print(".2f")
        except Exception as e:
            print(f"❌ Lỗi predict {name.upper()}: {e}")
            predictions[name] = None

    # Lưu vào log
    log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'prediction_log.csv')

    # Tạo log entry
    log_entry = {
        'date': pred_date,
        'rf_pred': predictions.get('rf'),
        'xgb_pred': predictions.get('xgb'),
        'lr_pred': predictions.get('lr'),
        'actual': None,  # Sẽ cập nhật sau
        'error_rf': None,
        'error_xgb': None,
        'error_lr': None,
        'predicted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Đọc log hiện tại hoặc tạo mới
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
    else:
        log_df = pd.DataFrame()

    # Thêm entry mới
    new_log = pd.DataFrame([log_entry])
    log_df = pd.concat([log_df, new_log], ignore_index=True)

    # Lưu
    log_df.to_csv(log_path, index=False)
    print(f"✓ Đã lưu prediction log: {log_path}")

    return predictions

if __name__ == "__main__":
    predict_daily()