# -*- coding: utf-8 -*-
"""
Tạo feature từ daily_combined để dự đoán lượng mưa **ngày mai** (target = rainfall.shift(-1)).
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from . import config


def _engineer_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    df["trend"] = (df["year"] - config.BASE_YEAR) * 365 + df["day_of_year"]

    for lag in [1, 2, 7, 14, 30]:
        df[f"rainfall_lag_{lag}"] = df["rainfall"].shift(lag)
    for w in [3, 7, 14]:
        df[f"rainfall_ma_{w}"] = df["rainfall"].rolling(w, min_periods=1).mean().shift(1)

    weather_cols = ["temperature", "humidity", "wind_speed"]
    if "cloud_cover" in df.columns:
        weather_cols.append("cloud_cover")
    if "surface_pressure" in df.columns:
        weather_cols.append("surface_pressure")

    for col in weather_cols:
        if col in df.columns:
            df[f"{col}_lag_1"] = df[col].shift(1)
            df[f"{col}_ma_7"] = df[col].rolling(7, min_periods=1).mean().shift(1)

    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity_ratio"] = df["temperature"] / (df["humidity"] + 1)
        df["temp_humidity_ratio_lag_1"] = df["temp_humidity_ratio"].shift(1)

    df["target"] = df["rainfall"].shift(-1)
    return df


def _feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"date", "rainfall", "target"}
    return [c for c in df.columns if c not in exclude]


def build_training_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Dataframe phục vụ train: chỉ các dòng có target (đã biết mưa ngày mai).
    """
    df = _engineer_columns(df)
    fc = _feature_columns(df)
    df = df.dropna(subset=fc + ["rainfall", "target"]).reset_index(drop=True)
    return df, fc


def build_for_prediction(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], pd.Timestamp]:
    """
    Giữ dòng cuối kể cả target=NaN (chưa có mưa ngày mai). Chỉ drop NaN ở feature.
    """
    df = _engineer_columns(df)
    fc = _feature_columns(df)
    df = df.dropna(subset=fc + ["rainfall"]).reset_index(drop=True)
    last_date = df["date"].iloc[-1]
    return df, fc, pd.Timestamp(last_date)


def last_row_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Ma trận 1 hàng — feature của ngày cuối cùng để dự đoán mưa ngày kế."""
    return df.iloc[-1][feature_cols].values.astype(np.float64).reshape(1, -1)


def load_daily_combined(path: str | None = None) -> pd.DataFrame:
    import os
    p = path or config.DAILY_COMBINED
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {p}")
    return pd.read_csv(p)


def merge_actuals_from_log(df: pd.DataFrame, log_path: str) -> pd.DataFrame:
    """Ghi đè rainfall theo actual trong log (nếu có) trước khi train."""
    import os
    if not os.path.exists(log_path):
        return df
    log = pd.read_csv(log_path)
    if "target_date" not in log.columns or "actual" not in log.columns:
        return df
    log["target_date"] = pd.to_datetime(log["target_date"]).dt.normalize()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for _, row in log.dropna(subset=["actual"]).iterrows():
        m = df["date"] == row["target_date"]
        if m.any():
            df.loc[m, "rainfall"] = float(row["actual"])
    return df
