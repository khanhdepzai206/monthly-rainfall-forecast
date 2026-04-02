#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chuẩn bị dữ liệu hàng ngày cho dự đoán lượng mưa ngày mai.
Tạo lag features từ 7 ngày trước.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def prepare_daily_data():
    """Chuẩn bị dữ liệu với lag features cho dự đoán hàng ngày."""

    # Đường dẫn
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Đọc dữ liệu mưa
    rain_df = pd.read_csv(os.path.join(data_dir, 'raw_daily.csv'), skiprows=2, header=None)
    rain_df.columns = ['datetime', 'rainfall']
    rain_df['date'] = pd.to_datetime(rain_df['datetime']).dt.date
    rain_df = rain_df.groupby('date')['rainfall'].sum().reset_index()

    # Đọc dữ liệu thời tiết
    weather_df = pd.read_csv(os.path.join(data_dir, 'weather_daily.csv'))
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date

    # Merge dữ liệu
    df = pd.merge(rain_df, weather_df, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)

    # Tạo lag features (7 ngày trước)
    lag_days = 7
    feature_cols = ['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'surface_pressure', 'rainfall']

    for col in feature_cols:
        for lag in range(1, lag_days + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Target: lượng mưa ngày mai
    df['target'] = df['rainfall'].shift(-1)

    # Loại bỏ NaN
    df = df.dropna().reset_index(drop=True)

    # Lưu dữ liệu đã chuẩn bị
    output_path = os.path.join(data_dir, 'daily_features.csv')
    df.to_csv(output_path, index=False)

    print(f"✓ Đã chuẩn bị dữ liệu: {len(df)} mẫu, lưu tại {output_path}")
    print(f"Features: {len([c for c in df.columns if 'lag' in c])} lag features")
    print(f"Target: rainfall ngày mai")

    return df

if __name__ == "__main__":
    prepare_daily_data()