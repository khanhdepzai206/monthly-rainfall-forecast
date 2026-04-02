# -*- coding: utf-8 -*-
"""
Train 3 mô hình dự đoán lượng mưa theo NGÀY - 2 giai đoạn:
  Giai đoạn 1: Phân loại - Có mưa hay không (rainfall > 0)
  Giai đoạn 2: Hồi quy - Lượng mưa (mm) khi có mưa

3 mô hình: Gradient Boosting, Random Forest, Extra Trees
"""
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = "../data/daily_combined.csv"
MODEL_DIR = "../models/"
BASE_YEAR = 1979


def build_features(df):
    """Tạo features từ daily_combined."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["trend"] = (df["year"] - BASE_YEAR) * 365 + df["day_of_year"]

    for lag in [1, 2, 7, 14, 30]:
        df[f"rainfall_lag_{lag}"] = df["rainfall"].shift(lag)
    for w in [3, 7, 14]:
        df[f"rainfall_ma_{w}"] = df["rainfall"].rolling(w, min_periods=1).mean().shift(1)

    for col in ["temperature", "humidity", "wind_speed"]:
        if col in df.columns:
            df[f"{col}_lag_1"] = df[col].shift(1)
            df[f"{col}_ma_7"] = df[col].rolling(7, min_periods=1).mean().shift(1)
    if "cloud_cover" in df.columns:
        df["cloud_cover_lag_1"] = df["cloud_cover"].shift(1)
    if "surface_pressure" in df.columns:
        df["surface_pressure_lag_1"] = df["surface_pressure"].shift(1)

    return df


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Chưa có {DATA_PATH}. Chạy prepare_data trước.")
        return

    df = pd.read_csv(DATA_PATH)
    df = build_features(df)
    df = df.dropna()

    exclude = ["date", "rainfall"]
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols]
    y_rainfall = df["rainfall"]
    y_has_rain = (y_rainfall > 0).astype(int)

    X_train, X_test, y_train_cls, y_test_cls = train_test_split(
        X, y_has_rain, test_size=0.2, random_state=42, shuffle=False
    )
    _, _, y_train_reg, y_test_reg = train_test_split(
        X, y_rainfall, test_size=0.2, random_state=42, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base_year = int(df["year"].min())
    all_metrics = {}

    def train_model(name, clf, reg):
        clf.fit(X_train_s, y_train_cls)
        y_pred_cls = clf.predict(X_test_s)
        acc = accuracy_score(y_test_cls, y_pred_cls)
        f1 = f1_score(y_test_cls, y_pred_cls, zero_division=0)
        prec = precision_score(y_test_cls, y_pred_cls, zero_division=0)
        rec = recall_score(y_test_cls, y_pred_cls, zero_division=0)

        mask_train_rain = y_train_cls == 1
        if mask_train_rain.sum() > 5:
            reg.fit(X_train_s[mask_train_rain], y_train_reg.values[mask_train_rain])
        else:
            reg.fit(X_train_s, y_train_reg.values)

        mask_test_rain = y_test_cls == 1
        if mask_test_rain.sum() > 0:
            X_rain = X_test_s[mask_test_rain]
            y_rain_true = y_test_reg.values[mask_test_rain]
            y_rain_pred = reg.predict(X_rain)
            y_rain_pred = np.maximum(y_rain_pred, 0)
            mae = mean_absolute_error(y_rain_true, y_rain_pred)
            rmse = np.sqrt(mean_squared_error(y_rain_true, y_rain_pred))
            r2 = r2_score(y_rain_true, y_rain_pred)
        else:
            mae = rmse = r2 = 0.0

        path = os.path.join(MODEL_DIR, f"daily_two_stage_{name.lower().replace(' ', '_')}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "classifier": clf,
                "regressor": reg,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "base_year": base_year,
                "metrics": {
                    "cls_accuracy": float(acc),
                    "cls_f1": float(f1),
                    "cls_precision": float(prec),
                    "cls_recall": float(rec),
                    "reg_mae": float(mae),
                    "reg_rmse": float(rmse),
                    "reg_r2": float(r2),
                },
            }, f)
        print(f"{name}: Accuracy={acc:.2%}, F1={f1:.3f} | Reg MAE={mae:.2f}, R²={r2:.4f}")
        return {"cls_accuracy": acc, "cls_f1": f1, "reg_mae": mae, "reg_r2": r2}

    # Model 1: Gradient Boosting (mạnh hơn để tăng R² / độ chính xác)
    train_model(
        "gradient_boosting",
        GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08, subsample=0.85, random_state=42
        ),
        GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.06, subsample=0.85, random_state=42
        ),
    )

    # Model 2: Random Forest
    train_model(
        "random_forest",
        RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
        RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
    )

    # Model 3: Extra Trees
    m3 = train_model(
        "extra_trees",
        ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
        ExtraTreesRegressor(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
    )

    # Load and save comparison metrics
    metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            mj = json.load(f)
    else:
        mj = {}

    for name in ["gradient_boosting", "random_forest", "extra_trees"]:
        p = os.path.join(MODEL_DIR, f"daily_two_stage_{name}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                d = pickle.load(f)
            mj[f"daily_two_stage_{name}"] = d.get("metrics", {})

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(mj, f, indent=2)
    print("\nSaved 3 two-stage models and model_metrics.json")


if __name__ == "__main__":
    main()
