#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cập nhật dữ liệu từ 2014 đến tháng 4/2026.
Fetch dữ liệu thời tiết từ Open-Meteo API và tạo lag features.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Thêm DuBao/src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DuBao', 'src'))

from fetch_weather_data import fetch_weather_data

def update_daily_features():
    """Cập nhật daily_features.csv với dữ liệu 2014-2026."""
    
    print("=" * 60)
    print("📊 CẬP NHẬT DỮ LIỆU LƯỢNG MƯA HÀNG NGÀY (2014-2026)")
    print("=" * 60)
    
    data_dir = os.path.join(os.path.dirname(__file__), 'DuBao', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Đọc dữ liệu cũ
    print("\n1️⃣  Đang đọc dữ liệu cũ (1979-2013)...")
    old_features_path = os.path.join(data_dir, 'daily_features.csv')
    df_old = pd.read_csv(old_features_path)
    df_old['date'] = pd.to_datetime(df_old['date']).dt.date
    print(f"   ✓ Đã tải: {len(df_old)} mẫu")
    print(f"   Khoảng: {df_old['date'].min()} đến {df_old['date'].max()}")
    
    # 2. Tìm ngày cuối cùng trong dữ liệu cũ
    last_date_old = pd.to_datetime(df_old['date'].max())
    start_fetch = (last_date_old + timedelta(days=1)).strftime('%Y-%m-%d')
    end_fetch = datetime(2026, 4, 2).strftime('%Y-%m-%d')
    
    print(f"\n2️⃣  Đang fetch dữ liệu thời tiết từ {start_fetch} đến {end_fetch}...")
    print("   (Gọi Open-Meteo API - miễn phí, không cần key)")
    
    try:
        weather_new = fetch_weather_data(start_fetch, end_fetch)
        if weather_new is None:
            raise Exception("API không trả về dữ liệu")
        print(f"   ✓ Đã fetch: {len(weather_new)} ngày")
    except Exception as e:
        print(f"   ❌ Lỗi fetch: {e}")
        print("   💡 Kiểm tra kết nối Internet hoặc API Open-Meteo")
        return False
    
    # 3. Đọc dữ liệu mưa cũ
    print("\n3️⃣  Đang xử lý dữ liệu mưa...")
    raw_daily_path = os.path.join(data_dir, 'raw_daily.csv')
    if os.path.exists(raw_daily_path):
        rain_df = pd.read_csv(raw_daily_path, skiprows=2, header=None)
        rain_df.columns = ['datetime', 'rainfall']
        rain_df['date'] = pd.to_datetime(rain_df['datetime']).dt.date
        rain_df = rain_df.groupby('date')['rainfall'].sum().reset_index()
        print(f"   ✓ Dữ liệu mưa lịch sử: {len(rain_df)} ngày")
    else:
        rain_df = pd.DataFrame(columns=['date', 'rainfall'])
        print(f"   ⚠️  Không tìm thấy raw_daily.csv - sử dụng dữ liệu trống")
    
    # 4. Ghép dữ liệu thời tiết mới với mưa
    print("\n4️⃣  Đang ghép dữ liệu...")
    weather_new['date'] = pd.to_datetime(weather_new['date']).dt.date
    df_new = pd.merge(weather_new, rain_df, on='date', how='left')
    
    # Nếu không có dữ liệu mưa, set mặc định = 0
    df_new['rainfall'] = df_new['rainfall'].fillna(0)
    
    print(f"   ✓ Dữ liệu mới: {len(df_new)} ngày")
    print(f"   Khoảng: {df_new['date'].min()} đến {df_new['date'].max()}")
    
    # 5. Ghép dữ liệu cũ + mới
    print("\n5️⃣  Đang ghép dữ liệu cũ + mới...")
    df_combined = pd.concat([
        df_old[['date', 'temperature', 'humidity', 'wind_speed', 'cloud_cover', 'surface_pressure', 'rainfall']],
        df_new[['date', 'temperature', 'humidity', 'wind_speed', 'cloud_cover', 'surface_pressure', 'rainfall']]
    ], ignore_index=True)
    
    df_combined = df_combined.drop_duplicates(subset=['date'], keep='last')
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    df_combined['date'] = pd.to_datetime(df_combined['date'])
    
    print(f"   ✓ Tổng cộng: {len(df_combined)} ngày")
    print(f"   Khoảng: {df_combined['date'].min().date()} đến {df_combined['date'].max().date()}")
    
    # 6. Tạo lag features (7 ngày)
    print("\n6️⃣  Đang tạo lag features...")
    lag_days = 7
    feature_cols = ['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'surface_pressure', 'rainfall']
    
    for col in feature_cols:
        for lag in range(1, lag_days + 1):
            df_combined[f'{col}_lag_{lag}'] = df_combined[col].shift(lag)
    
    # Target: lượng mưa ngày mai
    df_combined['target'] = df_combined['rainfall'].shift(-1)
    
    # Loại bỏ NaN (đặc biệt là những dòng không đủ 7 ngày lag)
    df_ready = df_combined.dropna().reset_index(drop=True)
    print(f"   ✓ Sau khi tạo lag features: {len(df_ready)} mẫu (xóa {len(df_combined) - len(df_ready)} NaN)")
    
    # 7. Lưu dữ liệu đã chuẩn bị
    print("\n7️⃣  Đang lưu dữ liệu...")
    output_path = os.path.join(data_dir, 'daily_features.csv')
    df_ready.to_csv(output_path, index=False)
    print(f"   ✓ Đã lưu: {output_path}")
    
    # 8. Lưu thống kê
    print("\n8️⃣  Thống kê:")
    print(f"   • Tổng mẫu: {len(df_ready)}")
    print(f"   • Lag features: {len([c for c in df_ready.columns if 'lag' in c])}")
    print(f"   • Khoảng dữ liệu: {df_ready['date'].min().date()} → {df_ready['date'].max().date()}")
    print(f"   • Năm có sẵn: {sorted(df_ready['date'].dt.year.unique())}")
    
    print("\n✅ CẬP NHẬT THÀNH CÔNG!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = update_daily_features()
    sys.exit(0 if success else 1)
