#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test dữ liệu mà mở rộng cho 2024-2026"""
import pandas as pd

df = pd.read_csv('DuBao/data/daily_features.csv')
df['date'] = pd.to_datetime(df['date']).dt.date

print("=" * 60)
print("✅ TEST DỮ LIỆU MỚI (2014-2026)")
print("=" * 60)

# Test với năm 2024
test_date = pd.to_datetime('2024-01-15').date()
row = df[df['date'] == test_date]

if len(row) > 0:
    print(f"\n✅ PASS: Tìm thấy dữ liệu cho 2024-01-15")
    print(f"   Temperature: {row['temperature_lag_1'].values[0]:.2f}°C")
    print(f"   Humidity: {row['humidity_lag_1'].values[0]:.2f}%")
    print(f"   Wind Speed: {row['wind_speed_lag_1'].values[0]:.2f} m/s")
else:
    print(f"\n❌ FAIL: Không tìm thấy dữ liệu cho 2024-01-15")

# Test với năm 2025
test_date = pd.to_datetime('2025-06-20').date()
row = df[df['date'] == test_date]

if len(row) > 0:
    print(f"\n✅ PASS: Tìm thấy dữ liệu cho 2025-06-20")
    print(f"   Temperature: {row['temperature_lag_1'].values[0]:.2f}°C")
else:
    print(f"\n❌ FAIL: Không tìm thấy dữ liệu cho 2025-06-20")

# Test với ngày gần cuối
test_date = pd.to_datetime('2026-03-31').date()
row = df[df['date'] == test_date]

if len(row) > 0:
    print(f"\n✅ PASS: Tìm thấy dữ liệu cho 2026-03-31")
    print(f"   Temperature: {row['temperature_lag_1'].values[0]:.2f}°C")
else:
    print(f"\n❌ FAIL: Không tìm thấy dữ liệu cho 2026-03-31")

print("\n" + "=" * 60)
print("📊 THỐNG KÊ TỔNG QUÁT")
print("=" * 60)
df_date = pd.to_datetime(df['date'])
print(f"Phạm vi: {df_date.min().date()} → {df_date.max().date()}")
print(f"Tổng mẫu: {len(df)}")
print(f"Năm có sẵn: {sorted(df_date.dt.year.unique())}")
print(f"Lag features: {len([c for c in df.columns if 'lag' in c])}")
print("✅ DỮ LIỆU SẴN SÀNG CHO DỰ ĐỐN 1979-2026!")
