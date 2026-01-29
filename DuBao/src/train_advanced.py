# -*- coding: utf-8 -*-
"""
Cải thiện độ chính xác R² cho mô hình dự đoán rainfall
Sử dụng kỹ thuật nâng cao: Stacking, Feature Selection, Cross-validation
"""
import pandas as pd
import numpy as np
import pickle
import os
import json
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression

DATA_PATH = "../data/daily_combined.csv"
MODEL_DIR = "../models/"
BASE_YEAR = 1979

def advanced_feature_engineering(df):
    """Tạo features nâng cao"""
    # Basic features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear

    # Seasonal features
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Trend
    df["trend"] = (df["year"] - BASE_YEAR) * 365 + df["day_of_year"]

    # Rainfall lags (extended)
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f"rainfall_lag_{lag}"] = df["rainfall"].shift(lag)

    # Rainfall rolling statistics
    for window in [3, 7, 14, 30]:
        df[f"rainfall_ma_{window}"] = df["rainfall"].rolling(window, min_periods=1).mean().shift(1)
        df[f"rainfall_std_{window}"] = df["rainfall"].rolling(window, min_periods=1).std().shift(1)
        df[f"rainfall_min_{window}"] = df["rainfall"].rolling(window, min_periods=1).min().shift(1)
        df[f"rainfall_max_{window}"] = df["rainfall"].rolling(window, min_periods=1).max().shift(1)

    # Weather features
    weather_cols = ["temperature", "humidity", "wind_speed"]
    if "cloud_cover" in df.columns:
        weather_cols.append("cloud_cover")
    if "surface_pressure" in df.columns:
        weather_cols.append("surface_pressure")

    for col in weather_cols:
        if col in df.columns:
            # Lags
            for lag in [1, 3, 7]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

            # Rolling stats
            for window in [3, 7]:
                df[f"{col}_ma_{window}"] = df[col].rolling(window, min_periods=1).mean().shift(1)
                df[f"{col}_std_{window}"] = df[col].rolling(window, min_periods=1).std().shift(1)

    # Advanced interactions
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity_ratio"] = df["temperature"] / (df["humidity"] + 1)
        df["temp_humidity_interaction"] = df["temperature"] * df["humidity"] / 100
        df["temp_humidity_ratio_lag_1"] = df["temp_humidity_ratio"].shift(1)

    if "cloud_cover" in df.columns and "humidity" in df.columns:
        df["cloud_humidity"] = df["cloud_cover"] * df["humidity"] / 100
        df["cloud_humidity_lag_1"] = df["cloud_humidity"].shift(1)

    if "wind_speed" in df.columns and "surface_pressure" in df.columns:
        df["wind_pressure_ratio"] = df["wind_speed"] / (df["surface_pressure"] / 1000)
        df["wind_pressure_ratio_lag_1"] = df["wind_pressure_ratio"].shift(1)

    # Rainfall patterns
    df["rainfall_change_1"] = df["rainfall"] - df["rainfall"].shift(1)
    df["rainfall_change_7"] = df["rainfall"] - df["rainfall"].shift(7)
    df["rainfall_momentum"] = df["rainfall_ma_3"] - df["rainfall_ma_7"]

    return df

def train_stacking_model(X_train_s, X_test_s, y_train, y_test, scaler, feature_cols, base_year):
    """Train stacking ensemble model"""
    # Base models
    base_models = [
        ('gb', GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )),
        ('xgb', xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
        ))
    ]

    # Meta model
    meta_model = Ridge(alpha=0.1)

    # Stacking model
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )

    print("Training Stacking Ensemble...")
    stacking_model.fit(X_train_s, y_train)
    y_pred = stacking_model.predict(X_test_s)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Stacking Ensemble - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "daily_stacking_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": stacking_model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "base_year": base_year,
            "metrics": {"mae": mae, "rmse": rmse, "r2_score": r2},
        }, f)

    return mae, rmse, r2

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Chưa có {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Advanced feature engineering
    df = advanced_feature_engineering(df)
    df = df.dropna()

    exclude = ["date", "rainfall"]
    feature_cols = [c for c in df.columns if c not in exclude]

    print(f"Total features: {len(feature_cols)}")

    # Feature selection - chọn 50 features tốt nhất
    X_temp = df[feature_cols]
    y_temp = df["rainfall"]

    selector = SelectKBest(score_func=f_regression, k=50)
    X_selected = selector.fit_transform(X_temp, y_temp)
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]

    print(f"Selected {len(selected_features)} best features")

    X = df[selected_features]
    y = df["rainfall"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train stacking model
    mae, rmse, r2 = train_stacking_model(X_train_s, X_test_s, y_train, y_test, scaler, selected_features, BASE_YEAR)

    # Cross-validation score
    cv_scores = cross_val_score(
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        X_train_s, y_train, cv=5, scoring='r2'
    )
    print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Update metrics
    metrics_path = "../models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            mj = json.load(f)
    else:
        mj = {}

    mj["daily_stacking"] = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4),
        "accuracy_percent": round(min(100, max(0, r2 * 100)), 1),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(mj, f, indent=2)

    print("✅ Đã cập nhật model_metrics.json với Stacking model")

if __name__ == "__main__":
    main()