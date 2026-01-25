import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

    # Model 1: Gradient Boosting với Weather Data
    print("Training Gradient Boosting với Weather Data...")
    gb_weather = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
    gb_weather.fit(X_train_scaled, y_train)
    y_pred_gb_weather = gb_weather.predict(X_test_scaled)

    mae_gb_weather = mean_absolute_error(y_test, y_pred_gb_weather)
    rmse_gb_weather = np.sqrt(mean_squared_error(y_test, y_pred_gb_weather))

    # Save model
    with open('../models/rainfall_model.pkl', 'wb') as f:
        pickle.dump({
            'model': gb_weather,
            'scaler': scaler,
            'feature_cols': feature_cols_combined,
            'metrics': {'mae': mae_gb_weather, 'rmse': rmse_gb_weather}
        }, f)

    # Model 2: Random Forest với Weather Data
    print("Training Random Forest với Weather Data...")
    rf_weather = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    rf_weather.fit(X_train_scaled, y_train)
    y_pred_rf_weather = rf_weather.predict(X_test_scaled)

    mae_rf_weather = mean_absolute_error(y_test, y_pred_rf_weather)
    rmse_rf_weather = np.sqrt(mean_squared_error(y_test, y_pred_rf_weather))

    # Save model
    with open('../models/rainfall_model_rf.pkl', 'wb') as f:
        pickle.dump({
            'model': rf_weather,
            'scaler': scaler,
            'feature_cols': feature_cols_combined,
            'metrics': {'mae': mae_rf_weather, 'rmse': rmse_rf_weather}
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

        # Save SARIMA model
        with open('../models/sarimax_model.pkl', 'wb') as f:
            pickle.dump({
                'model': sarima_fit,
                'metrics': {'mae': mae_sarima, 'rmse': rmse_sarima, 'aic': sarima_fit.aic}
            }, f)

    except Exception as e:
        print(f"SARIMA training failed: {e}")
        mae_sarima = rmse_sarima = None

    # Print results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS (TẤT CẢ ĐỀU SỬ DỤNG WEATHER DATA):")
    print("="*60)
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    if mae_sarima:
        print(".2f")
        print(".2f")

if __name__ == "__main__":
    evaluate_models()