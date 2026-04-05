#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chuẩn bị dữ liệu hàng ngày cho dự đoán lượng mưa ngày mai.
Tạo lag features từ 7 ngày trước.
"""
import pandas as pd
import numpy as np
import os

def prepare_daily_data():
    """Chuẩn bị dữ liệu với feature engineering để dự đoán lượng mưa ngày mai."""

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    rain_df = pd.read_csv(os.path.join(data_dir, 'raw_daily.csv'), skiprows=2, header=None)
    rain_df.columns = ['datetime', 'rainfall']
    rain_df['date'] = pd.to_datetime(rain_df['datetime']).dt.date
    rain_df = rain_df.groupby('date')['rainfall'].sum().reset_index()

    weather_df = pd.read_csv(os.path.join(data_dir, 'weather_daily.csv'))
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date

    df = pd.merge(rain_df, weather_df, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['is_wet'] = (df['rainfall'] > 0).astype(int)

    lag_days = 7
    feature_cols = ['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'surface_pressure', 'rainfall']
    for col in feature_cols:
        for lag in range(1, lag_days + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    df['rainfall_ma_3'] = df['rainfall'].rolling(3, min_periods=1).mean().shift(1)
    df['rainfall_ma_7'] = df['rainfall'].rolling(7, min_periods=1).mean().shift(1)
    df['wet_spell_3'] = df['is_wet'].rolling(3, min_periods=1).sum().shift(1)
    df['wet_spell_7'] = df['is_wet'].rolling(7, min_periods=1).sum().shift(1)
    df['temperature_ma_3'] = df['temperature'].rolling(3, min_periods=1).mean().shift(1)
    df['humidity_ma_3'] = df['humidity'].rolling(3, min_periods=1).mean().shift(1)

    df['target'] = df['rainfall'].shift(-1)
    df = df.dropna().reset_index(drop=True)

    output_path = os.path.join(data_dir, 'daily_features.csv')
    df.to_csv(output_path, index=False)

    feature_count = len([c for c in df.columns if c not in ['date', 'datetime', 'rainfall', 'target']])
    print(f"✓ Đã chuẩn bị dữ liệu: {len(df)} mẫu, lưu tại {output_path}")
    print(f"Features: {feature_count} features")
    return df

if __name__ == "__main__":
    prepare_daily_data()