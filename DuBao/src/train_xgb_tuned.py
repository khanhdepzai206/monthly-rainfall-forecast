# -*- coding: utf-8 -*-
"""
Cải thiện nhanh độ chính xác với XGBoost tuned
"""
import pandas as pd
import numpy as np
import pickle
import os
import json
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

DATA_PATH = "../data/daily_combined.csv"
MODEL_DIR = "../models/"
BASE_YEAR = 1979

def main():
    print("Starting XGBoost tuning...")
    if not os.path.exists(DATA_PATH):
        print(f"Chưa có {DATA_PATH}")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data shape: {df.shape}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Enhanced features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["trend"] = (df["year"] - BASE_YEAR) * 365 + df["day_of_year"]

    # Extended lags
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f"rainfall_lag_{lag}"] = df["rainfall"].shift(lag)

    # Rolling stats
    for window in [3, 7, 14]:
        df[f"rainfall_ma_{window}"] = df["rainfall"].rolling(window, min_periods=1).mean().shift(1)
        df[f"rainfall_std_{window}"] = df["rainfall"].rolling(window, min_periods=1).std().shift(1)

    # Weather features
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

    print(f"Features: {len(feature_cols)}")

    X = df[feature_cols]
    y = df["rainfall"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Grid search for best parameters
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [6, 8],
        'learning_rate': [0.05, 0.1],
        'min_child_weight': [3, 5],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }

    print("Grid search for best XGBoost parameters...")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_s, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

    # Train final model with best parameters
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_s)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Final XGBoost - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "daily_xgb_tuned_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "base_year": BASE_YEAR,
            "metrics": {"mae": mae, "rmse": rmse, "r2_score": r2},
            "best_params": grid_search.best_params_
        }, f)

    # Update metrics
    metrics_path = "../models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            mj = json.load(f)
    else:
        mj = {}

    mj["daily_xgb_tuned"] = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4),
        "accuracy_percent": round(min(100, max(0, r2 * 100)), 1),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(mj, f, indent=2)

    print("✅ Đã lưu XGBoost tuned model")

if __name__ == "__main__":
    main()