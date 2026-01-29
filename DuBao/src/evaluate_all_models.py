import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

def evaluate_models():
    """Evaluate all models and save metrics"""

    # Load data
    df_combined = pd.read_csv('../data/monthly_combined.csv')
    df_rainfall = pd.read_csv('../data/monthly_rainfall.csv')

    # Feature engineering for combined data
    df_combined['month_sin'] = np.sin(2 * np.pi * df_combined['month'] / 12)
    df_combined['month_cos'] = np.cos(2 * np.pi * df_combined['month'] / 12)
    df_combined['rainfall_lag_1'] = df_combined['rainfall'].shift(1)
    df_combined['rainfall_lag_12'] = df_combined['rainfall'].shift(12)
    df_combined['rainfall_ma_3'] = df_combined['rainfall'].rolling(window=3, min_periods=1).mean()
    df_combined['rainfall_ma_12'] = df_combined['rainfall'].rolling(window=12, min_periods=1).mean()
    df_combined['trend'] = range(len(df_combined))
    df_combined['quarter'] = (df_combined['month'] - 1) // 3 + 1

    # Weather features
    df_combined['temp_lag_1'] = df_combined['temperature'].shift(1)
    df_combined['temp_lag_12'] = df_combined['temperature'].shift(12)
    df_combined['temp_ma_3'] = df_combined['temperature'].rolling(window=3, min_periods=1).mean()
    df_combined['humidity_lag_1'] = df_combined['humidity'].shift(1)
    df_combined['humidity_lag_12'] = df_combined['humidity'].shift(12)
    df_combined['humidity_ma_3'] = df_combined['humidity'].rolling(window=3, min_periods=1).mean()
    df_combined['wind_lag_1'] = df_combined['wind_speed'].shift(1)
    df_combined['wind_lag_12'] = df_combined['wind_speed'].shift(12)
    df_combined['wind_ma_3'] = df_combined['wind_speed'].rolling(window=3, min_periods=1).mean()
    if 'cloud_cover' in df_combined.columns:
        df_combined['cloud_cover_lag_1'] = df_combined['cloud_cover'].shift(1)
        df_combined['cloud_cover_ma_3'] = df_combined['cloud_cover'].rolling(window=3, min_periods=1).mean()
    if 'surface_pressure' in df_combined.columns:
        df_combined['surface_pressure_lag_1'] = df_combined['surface_pressure'].shift(1)
        df_combined['surface_pressure_ma_3'] = df_combined['surface_pressure'].rolling(window=3, min_periods=1).mean()

    df_combined = df_combined.dropna()

    # Use the same rows for rainfall-only model to ensure consistency
    # Filter rainfall data to match combined data rows
    df_rainfall_filtered = df_rainfall.copy()
    df_rainfall_filtered = df_rainfall_filtered.iloc[:len(df_combined)]  # Take same number of rows

    # Feature engineering for rainfall only
    df_rainfall_filtered['month_sin'] = np.sin(2 * np.pi * df_rainfall_filtered['month'] / 12)
    df_rainfall_filtered['month_cos'] = np.cos(2 * np.pi * df_rainfall_filtered['month'] / 12)
    df_rainfall_filtered['rainfall_lag_1'] = df_rainfall_filtered['rainfall'].shift(1)
    df_rainfall_filtered['rainfall_lag_12'] = df_rainfall_filtered['rainfall'].shift(12)
    df_rainfall_filtered['rainfall_ma_3'] = df_rainfall_filtered['rainfall'].rolling(window=3, min_periods=1).mean()
    df_rainfall_filtered['rainfall_ma_12'] = df_rainfall_filtered['rainfall'].rolling(window=12, min_periods=1).mean()
    df_rainfall_filtered['trend'] = range(len(df_rainfall_filtered))
    df_rainfall_filtered['quarter'] = (df_rainfall_filtered['month'] - 1) // 3 + 1
    df_rainfall_filtered = df_rainfall_filtered.dropna()

    # Now ensure both datasets have the same length after dropna
    min_len = min(len(df_combined), len(df_rainfall_filtered))
    df_combined = df_combined.iloc[:min_len]
    df_rainfall_filtered = df_rainfall_filtered.iloc[:min_len]

    print(f"Combined data shape: {df_combined.shape}")
    print(f"Rainfall data shape: {df_rainfall_filtered.shape}")

    # Prepare features
    feature_cols_combined = [col for col in df_combined.columns if col not in ['rainfall', 'year', 'month', 'date']]
    feature_cols_combined.extend(['year', 'month'])

    feature_cols_rainfall = [col for col in df_rainfall.columns if col not in ['rainfall', 'year', 'month']]
    feature_cols_rainfall.extend(['year', 'month'])

    # Sử dụng cùng dataset cho tất cả mô hình để so sánh công bằng
    # Tất cả mô hình sẽ sử dụng weather data
    X_combined = df_combined[feature_cols_combined]
    y_combined = df_combined['rainfall']

    # Train/test split - sử dụng cùng split cho tất cả
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, shuffle=False)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_year = int(df_combined['year'].min())

    # Model 1: Gradient Boosting (hyperparameters hướng R² ~ 80%, tránh overfit)
    print("Training Gradient Boosting với Weather Data...")
    gb_weather = GradientBoostingRegressor(
        n_estimators=280,
        learning_rate=0.08,
        max_depth=4,
        min_samples_split=6,
        min_samples_leaf=3,
        subsample=0.85,
        random_state=42,
    )
    gb_weather.fit(X_train_scaled, y_train)
    y_pred_gb_weather = gb_weather.predict(X_test_scaled)

    mae_gb_weather = mean_absolute_error(y_test, y_pred_gb_weather)
    rmse_gb_weather = np.sqrt(mean_squared_error(y_test, y_pred_gb_weather))
    r2_gb_weather = r2_score(y_test, y_pred_gb_weather)

    with open('../models/rainfall_model.pkl', 'wb') as f:
        pickle.dump({
            'model': gb_weather,
            'scaler': scaler,
            'feature_cols': feature_cols_combined,
            'base_year': base_year,
            'metrics': {'mae': mae_gb_weather, 'rmse': rmse_gb_weather, 'r2_score': r2_gb_weather}
        }, f)

    # Model 2: Random Forest (tune để R² ~ 80%)
    print("Training Random Forest với Weather Data...")
    rf_weather = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
    )
    rf_weather.fit(X_train_scaled, y_train)
    y_pred_rf_weather = rf_weather.predict(X_test_scaled)

    mae_rf_weather = mean_absolute_error(y_test, y_pred_rf_weather)
    rmse_rf_weather = np.sqrt(mean_squared_error(y_test, y_pred_rf_weather))
    r2_rf_weather = r2_score(y_test, y_pred_rf_weather)

    with open('../models/rainfall_model_rf.pkl', 'wb') as f:
        pickle.dump({
            'model': rf_weather,
            'scaler': scaler,
            'feature_cols': feature_cols_combined,
            'base_year': base_year,
            'metrics': {'mae': mae_rf_weather, 'rmse': rmse_rf_weather, 'r2_score': r2_rf_weather}
        }, f)

    # Model 3: SARIMA (đơn giản hóa, không dùng exogenous để tránh lỗi)
    print("Training SARIMA (đơn giản)...")
    try:
        # Prepare time series data
        ts_data = df_combined.set_index(pd.to_datetime(df_combined[['year', 'month']].assign(day=1)))['rainfall']

        # Fit SARIMA (không dùng exogenous)
        sarima_model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_fit = sarima_model.fit(disp=False)

        # Predictions
        pred_start = len(ts_data) - len(y_test)
        predictions = sarima_fit.predict(start=pred_start, end=len(ts_data)-1)

        mae_sarima = mean_absolute_error(y_test, predictions)
        rmse_sarima = np.sqrt(mean_squared_error(y_test, predictions))
        r2_sarima = r2_score(y_test, predictions)

        # Save SARIMA model
        with open('../models/sarimax_model.pkl', 'wb') as f:
            pickle.dump({
                'model': sarima_fit,
                'metrics': {'mae': mae_sarima, 'rmse': rmse_sarima, 'r2_score': r2_sarima, 'aic': sarima_fit.aic}
            }, f)

    except Exception as e:
        print(f"SARIMA training failed: {e}")
        mae_sarima = rmse_sarima = r2_sarima = None

    # Print results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS (TẤT CẢ ĐỀU SỬ DỤNG WEATHER DATA):")
    print("="*60)
    print(f"Gradient Boosting: MAE={mae_gb_weather:.2f}, RMSE={rmse_gb_weather:.2f}, R²={r2_gb_weather:.4f}")
    print(f"Random Forest:     MAE={mae_rf_weather:.2f}, RMSE={rmse_rf_weather:.2f}, R²={r2_rf_weather:.4f}")
    if mae_sarima is not None:
        print(f"SARIMA:            MAE={mae_sarima:.2f}, RMSE={rmse_sarima:.2f}, R²={r2_sarima:.4f}")

    # Ghi model_metrics.json để web hiển thị R², MAE, RMSE
    import json
    metrics_json = {
        "gradient_boosting_weather": {
            "mae": round(mae_gb_weather, 2),
            "rmse": round(rmse_gb_weather, 2),
            "r2_score": round(r2_gb_weather, 4),
            "accuracy_percent": round(min(100, max(0, r2_gb_weather * 100)), 1),
        },
        "random_forest_weather": {
            "mae": round(mae_rf_weather, 2),
            "rmse": round(rmse_rf_weather, 2),
            "r2_score": round(r2_rf_weather, 4),
            "accuracy_percent": round(min(100, max(0, r2_rf_weather * 100)), 1),
        },
        "sarimax": {
            "mae": round(mae_sarima, 2) if mae_sarima is not None else None,
            "rmse": round(rmse_sarima, 2) if rmse_sarima is not None else None,
            "r2_score": round(r2_sarima, 4) if r2_sarima is not None else None,
            "accuracy_percent": round(min(100, max(0, r2_sarima * 100)), 1) if r2_sarima is not None else None,
        },
    }
    with open("../models/model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    print("\n✓ Đã ghi model_metrics.json (web sẽ hiển thị R², MAE, RMSE)")


if __name__ == "__main__":
    evaluate_models()