# -*- coding: utf-8 -*-
"""
Train Ensemble model để đạt R² cao hơn
"""
import pandas as pd
import numpy as np
import pickle
import os
import json
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "../data/daily_combined.csv"
MODEL_DIR = "../models/"
BASE_YEAR = 1979

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Chưa có {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Features tối ưu
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["trend"] = (df["year"] - BASE_YEAR) * 365 + df["day_of_year"]

    # Lags rainfall
    for lag in [1, 3, 7, 14]:
        df[f"rainfall_lag_{lag}"] = df["rainfall"].shift(lag)
    
    # Moving averages
    for window in [3, 7, 14]:
        df[f"rainfall_ma_{window}"] = df["rainfall"].rolling(window, min_periods=1).mean().shift(1)

    # Weather features
    weather_cols = ["temperature", "humidity", "wind_speed", "cloud_cover", "surface_pressure"]
    for col in weather_cols:
        if col in df.columns:
            df[f"{col}_lag_1"] = df[col].shift(1)
            df[f"{col}_ma_7"] = df[col].rolling(7, min_periods=1).mean().shift(1)

    # Interactions
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity_ratio"] = df["temperature"] / (df["humidity"] + 1)
        df["temp_humidity_ratio_lag"] = df["temp_humidity_ratio"].shift(1)

    df = df.dropna()
    exclude = ["date", "rainfall"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols]
    y = df["rainfall"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"Features: {len(feature_cols)}")

    # Create individual models
    gb = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
    )
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=42
    )
    xgb_model = xgb.XGBRegressor(
        n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
    )

    # Ensemble model
    ensemble = VotingRegressor([
        ('gb', gb),
        ('rf', rf), 
        ('xgb', xgb_model)
    ])

    print("Training Ensemble...")
    ensemble.fit(X_train_s, y_train)
    y_pred = ensemble.predict(X_test_s)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Ensemble - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "daily_ensemble_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": ensemble,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "base_year": BASE_YEAR,
            "metrics": {"mae": mae, "rmse": rmse, "r2_score": r2},
        }, f)

    # Update metrics
    metrics_path = "../models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            mj = json.load(f)
    else:
        mj = {}

    mj["daily_ensemble"] = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4),
        "accuracy_percent": round(min(100, max(0, r2 * 100)), 1),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(mj, f, indent=2)

    print("✅ Đã cập nhật model_metrics.json")

if __name__ == "__main__":
    main()