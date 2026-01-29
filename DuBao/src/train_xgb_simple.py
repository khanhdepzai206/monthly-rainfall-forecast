# -*- coding: utf-8 -*-
"""
Train mô hình XGBoost đơn giản để test độ chính xác cao hơn
"""
import pandas as pd
import numpy as np
import pickle
import os
import json
import xgboost as xgb
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
    df["trend"] = (df["year"] - BASE_YEAR) * 365 + df["day_of_year"]

    # Thêm features seasonal và interactions
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Interactions
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity"] = df["temperature"] * df["humidity"] / 100
        df["temp_humidity_lag"] = df["temp_humidity"].shift(1)
    
    if "cloud_cover" in df.columns and "humidity" in df.columns:
        df["cloud_humidity"] = df["cloud_cover"] * df["humidity"] / 100
        df["cloud_humidity_lag"] = df["cloud_humidity"].shift(1)
    
    # More rainfall lags
    df["rainfall_lag_3"] = df["rainfall"].shift(3)
    df["rainfall_lag_14"] = df["rainfall"].shift(14)
    df["rainfall_ma_3"] = df["rainfall"].rolling(3, min_periods=1).mean().shift(1)
    df["rainfall_ma_14"] = df["rainfall"].rolling(14, min_periods=1).mean().shift(1)

    # Weather features
    if "temperature" in df.columns:
        df["temp_lag_1"] = df["temperature"].shift(1)
    if "humidity" in df.columns:
        df["humidity_lag_1"] = df["humidity"].shift(1)
    if "wind_speed" in df.columns:
        df["wind_lag_1"] = df["wind_speed"].shift(1)
    if "cloud_cover" in df.columns:
        df["cloud_cover_lag_1"] = df["cloud_cover"].shift(1)
    if "surface_pressure" in df.columns:
        df["surface_pressure_lag_1"] = df["surface_pressure"].shift(1)

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

    print(f"Features: {len(feature_cols)} - {feature_cols[:5]}...")

    # XGBoost conservative tuning
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='reg:squarederror',
        random_state=42,
    )

    print("Training XGBoost with early stopping...")
    eval_set = [(X_train_s, y_train), (X_test_s, y_test)]
    xgb_model.fit(
        X_train_s, y_train,
        eval_set=eval_set,
        verbose=False
    )
    y_pred = xgb_model.predict(X_test_s)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"XGBoost - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "daily_xgboost_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": xgb_model,
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

    mj["daily_xgboost"] = {
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