# -*- coding: utf-8 -*-
"""
Train LSTM model cho dự đoán rainfall
"""
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = "../data/daily_combined.csv"
MODEL_DIR = "../models/"
BASE_YEAR = 1979

def create_sequences(X, y, seq_length=30):
    """Create sequences for LSTM"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Chưa có {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["trend"] = (df["year"] - BASE_YEAR) * 365 + df["day_of_year"]

    # Lags
    for lag in [1, 3, 7]:
        df[f"rainfall_lag_{lag}"] = df["rainfall"].shift(lag)
    
    # Weather
    weather_cols = ["temperature", "humidity", "wind_speed", "cloud_cover", "surface_pressure"]
    for col in weather_cols:
        if col in df.columns:
            df[f"{col}_lag_1"] = df[col].shift(1)

    df = df.dropna()
    exclude = ["date", "rainfall"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df["rainfall"].values

    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create sequences
    seq_length = 14  # 14 days lookback
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    
    print(f"Sequences shape: {X_seq.shape}, Target shape: {y_seq.shape}")

    # Train/test split
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, input_shape=(seq_length, X_seq.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("Training LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Predictions
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)
    print(f"LSTM - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "daily_lstm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "feature_cols": feature_cols,
            "base_year": BASE_YEAR,
            "seq_length": seq_length,
            "metrics": {"mae": mae, "rmse": rmse, "r2_score": r2},
        }, f)

    # Update metrics
    metrics_path = "../models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            mj = json.load(f)
    else:
        mj = {}

    mj["daily_lstm"] = {
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