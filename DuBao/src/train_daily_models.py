#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train 3 mô hình Machine Learning cho dự đoán lượng mưa hàng ngày.
Models: RandomForest, XGBoost, LinearRegression
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import pickle
import os
from datetime import datetime

def train_daily_models():
    """Train và lưu 3 mô hình."""

    # Đọc dữ liệu đã chuẩn bị
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    try:
        df = pd.read_csv(os.path.join(data_dir, 'daily_features.csv'))
        print(f"✓ Loaded data: {len(df)} samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Features và target
    feature_cols = [c for c in df.columns if 'lag' in c]
    X = df[feature_cols]
    y = df['target']

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    models = {
        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
        'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'lr': LinearRegression()
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name.upper()} ---")

        # Train
        model.fit(X_train, y_train)

        # Predict trên test
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse
        }

        print(".4f")
        print(".4f")

        # Lưu model
        model_path = os.path.join(models_dir, f'{name}_daily_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Đã lưu model: {model_path}")

    # Lưu metrics
    metrics_df = pd.DataFrame({
        'model': list(results.keys()),
        'mae': [r['mae'] for r in results.values()],
        'rmse': [r['rmse'] for r in results.values()],
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

    metrics_path = os.path.join(models_dir, 'daily_model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Đã lưu metrics: {metrics_path}")

    # Tìm model tốt nhất
    best_model = min(results.items(), key=lambda x: x[1]['mae'])
    print(f"\n🏆 Model tốt nhất: {best_model[0].upper()} (MAE: {best_model[1]['mae']:.4f})")

    return results

if __name__ == "__main__":
    train_daily_models()