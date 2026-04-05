#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train 3 mô hình Machine Learning cho dự đoán lượng mưa hàng ngày.
Models: RandomForest, XGBoost, LinearRegression
"""
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def _get_feature_columns(df):
    exclude = {'date', 'target', 'datetime', 'rainfall'}
    return [c for c in df.columns if c not in exclude]


def _build_models(random_state=42):
    return {
        'rf': TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        'lr': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LinearRegression()),
            ]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        'xgb': TransformedTargetRegressor(
            regressor=xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_lambda=2.0,
                min_child_weight=3,
                objective='reg:squarederror',
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }

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

    feature_cols = _get_feature_columns(df)
    X = df[feature_cols]
    y = df['target'].clip(lower=0.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    models = _build_models(random_state=42)

    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name.upper()} ---")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0.0, None)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
        }

        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        model_path = os.path.join(models_dir, f'{name}_daily_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Đã lưu model: {model_path}")

    metrics_df = pd.DataFrame({
        'model': list(results.keys()),
        'mae': [r['mae'] for r in results.values()],
        'rmse': [r['rmse'] for r in results.values()],
        'train_date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * len(results),
    })

    metrics_path = os.path.join(models_dir, 'daily_model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Đã lưu metrics: {metrics_path}")

    best_model = min(results.items(), key=lambda x: x[1]['mae'])
    print(f"\n🏆 Model tốt nhất: {best_model[0].upper()} (MAE: {best_model[1]['mae']:.4f})")

    return results

if __name__ == "__main__":
    train_daily_models()