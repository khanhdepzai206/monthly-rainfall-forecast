# -*- coding: utf-8 -*-
"""
Train 3 mô hình dự đoán lượng mưa theo NGÀY: Gradient Boosting, Random Forest, và Linear Regression.
Dữ liệu: daily_combined.csv (date, rainfall, temperature, humidity, wind_speed, ...)
"""
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "../data/daily_combined.csv"
MODEL_DIR = "../models/"
BASE_YEAR = 1979

def train_and_save_model(model, model_name, X_train_s, X_test_s, y_train, y_test, scaler, feature_cols, base_year):
    """Train model, evaluate, and save."""
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    model_path = os.path.join(MODEL_DIR, f"daily_{model_name.lower().replace(' ', '_')}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "base_year": base_year,
            "metrics": {"mae": mae, "rmse": rmse, "r2_score": r2},
        }, f)
    print(f"Đã lưu mô hình: {model_path}")
    return mae, rmse, r2

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Chưa có {DATA_PATH}. Chạy prepare_data_daily() trong preprocess trước.")
        return

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Feature theo ngày - tối ưu để tăng độ chính xác
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["trend"] = (df["year"] - BASE_YEAR) * 365 + df["day_of_year"]
    
    # Lags cho rainfall
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"rainfall_lag_{lag}"] = df["rainfall"].shift(lag)
    
    # Moving averages cho rainfall
    for window in [3, 7, 14]:
        df[f"rainfall_ma_{window}"] = df["rainfall"].rolling(window, min_periods=1).mean().shift(1)
    
    # Features thời tiết
    weather_cols = ["temperature", "humidity", "wind_speed"]
    if "cloud_cover" in df.columns:
        weather_cols.append("cloud_cover")
    if "surface_pressure" in df.columns:
        weather_cols.append("surface_pressure")
    
    for col in weather_cols:
        if col in df.columns:
            df[f"{col}_lag_1"] = df[col].shift(1)
            df[f"{col}_ma_7"] = df[col].rolling(7, min_periods=1).mean().shift(1)
    
    # Interactions
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity_ratio"] = df["temperature"] / (df["humidity"] + 1)
        df["temp_humidity_ratio_lag_1"] = df["temp_humidity_ratio"].shift(1)

    df = df.dropna()
    exclude = ["date", "rainfall"]
    feature_cols = [c for c in df.columns if c not in exclude]
    if "year" not in feature_cols:
        feature_cols.append("year")
    if "month" not in feature_cols:
        feature_cols.extend(["month", "day"])

    X = df[feature_cols]
    y = df["rainfall"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base_year = int(df["year"].min())

    # Model 1: Gradient Boosting - tuned for higher accuracy
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42,
    )
    mae_gb, rmse_gb, r2_gb = train_and_save_model(gb_model, "Gradient Boosting", X_train_s, X_test_s, y_train, y_test, scaler, feature_cols, base_year)

    # Model 2: Random Forest - tuned for higher accuracy
    rf_model = RandomForestRegressor(
        n_estimators=50,  # Giảm để tránh interrupt
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
    )
    mae_rf, rmse_rf, r2_rf = train_and_save_model(rf_model, "Random Forest", X_train_s, X_test_s, y_train, y_test, scaler, feature_cols, base_year)

    # Model 3: XGBoost - powerful gradient boosting
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,  # Giảm để tránh interrupt
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
    )
    mae_xgb, rmse_xgb, r2_xgb = train_and_save_model(xgb_model, "XGBoost", X_train_s, X_test_s, y_train, y_test, scaler, feature_cols, base_year)

    # Update model_metrics.json
    metrics_path = "../models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            mj = json.load(f)
    else:
        mj = {}

    mj["daily_gradient_boosting"] = {
        "mae": round(mae_gb, 2),
        "rmse": round(rmse_gb, 2),
        "r2_score": round(r2_gb, 4),
        "accuracy_percent": round(min(100, max(0, r2_gb * 100)), 1),
    }
    mj["daily_random_forest"] = {
        "mae": round(mae_rf, 2),
        "rmse": round(rmse_rf, 2),
        "r2_score": round(r2_rf, 4),
        "accuracy_percent": round(min(100, max(0, r2_rf * 100)), 1),
    }
    mj["daily_xgboost"] = {
        "mae": round(mae_xgb, 2),
        "rmse": round(rmse_xgb, 2),
        "r2_score": round(r2_xgb, 4),
        "accuracy_percent": round(min(100, max(0, r2_xgb * 100)), 1),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(mj, f, indent=2)
    print("Đã cập nhật model_metrics.json với 3 mô hình daily.")


if __name__ == "__main__":
    main()
