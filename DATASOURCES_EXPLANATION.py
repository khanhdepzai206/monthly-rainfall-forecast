#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Văn bản giải thích nguồn dữ liệu cho 2 loại dự đoán
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║       NGUỒN DỮ LIỆU CHO DỰ ĐOÁN NGÀY MAI + 7 NGÀY TIẾP THEO              ║
╚════════════════════════════════════════════════════════════════════════════╝

📅 TRANG 1: /daily-predict/ (DỰ ĐOÁN NGÀY MAI)
════════════════════════════════════════════════════════════════════════════

🔍 Dữ liệu đầu vào:
   ├─ Tệp: DuBao/data/daily_features.csv
   ├─ Dòng sử dụng: DỰA TRÊN HÔM NAY → Dự đoán NGÀY MAI (D+1)
   └─ 35 lag features từ 7 ngày gần nhất (T-1 đến T-7)
      • temperature_lag_1 ... temperature_lag_7
      • humidity_lag_1 ... humidity_lag_7
      • wind_speed_lag_1 ... wind_speed_lag_7
      • cloud_cover_lag_1 ... cloud_cover_lag_7
      • surface_pressure_lag_1 ... surface_pressure_lag_7
      • rainfall_lag_1 ... rainfall_lag_7

📊 Dữ liệu này từ đâu?
   1. Dữ liệu mưa: raw_daily.csv (dữ liệu nguyên bản)
   2. Dữ liệu thời tiết: weather_daily.csv
      └─ Fetch từ Open-Meteo API (lịch sử từ 1979-2026)
      └─ URL: https://archive-api.open-meteo.com/v1/archive
      └─ Tính năng: Nhiệt độ, độ ẩm, gió, mây, áp suất
   
   3. Xử lý:
      • Merge mưa + thời tiết theo ngày
      • Tạo lag features (dùng .shift() trong pandas)
      • Lưu vào daily_features.csv

⚙️ Mô hình dự đoán:
   ├─ RandomForest (rf_daily_model.pkl)
   ├─ XGBoost (xgb_daily_model.pkl)
   └─ LinearRegression (lr_daily_model.pkl)

✅ KẾT QUẢ:
   → Dự đoán 3 giá trị tuyệt đối: RF, XGB, LR (mm/ngày)
   → CHỈ là dự đoán cho NGÀY MAI (D+1)


════════════════════════════════════════════════════════════════════════════

📅 TRANG 2: /predict/ (DỰ ĐOÁN 6-7 NGÀY TIẾP THEO)
════════════════════════════════════════════════════════════════════════════

🔍 Dữ liệu đầu vào:
   ├─ Nguồn 1: DuBao/data/daily_features.csv
   │           (Dữ liệu lịch sử cuối cùng: 01/04/2026)
   │
   └─ Nguồn 2: Open-Meteo FORECAST API (thời tiết dự báo)
              URL: https://api.open-meteo.com/v1/forecast
              Cho phép: 7-10 ngày tiếp theo
              Tính năng: Nhiệt độ, độ ẩm, gió, mây, áp suất (dự báo)

📊 QUY TRÌNH ROLLING FORECAST:
   
   Ngày 01/04 (đã có dữ liệu thực tế):
   ├─ Lấy: daily_features.csv (dòng 01/04)
   ├─ Lag features: từ 25/03 - 01/04 ✓
   └─ Dự đoán: RF, XGB, LR cho 02/04

   Ngày 02/04 (DÙNG DỰ ĐOÁN):
   ├─ Fetch: Thời tiết dự báo cho 02/04
   ├─ Cập nhật lag features với:
   │   • Thời tiết dự báo: temperature_forecast, humidity_forecast, ...
   │   • Mưa dự đoán: rainfall_lag_1 = RF_pred từ 01/04
   ├─ Lag mới: shift(1) → temperature_lag_2..7 từ lag cũ
   └─ Dự đoán: RF, XGB, LR cho 03/04

   Ngày 03/04 (DÙNG DỰ ĐOÁN):
   ├─ Fetch: Thời tiết dự báo cho 03/04
   ├─ Cập nhật lag features với:
   │   • Thời tiết dự báo: 03/04
   │   • Mưa dự đoán: rainfall_lag_1 = RF_pred từ 02/04
   └─ Dự đoán: RF, XGB, LR cho 04/04
   
   ... (lặp 6 lần: 02/04 - 07/04)

⚙️ Mô hình:
   ├─ RandomForest (rf_daily_model.pkl)
   ├─ XGBoost (xgb_daily_model.pkl)
   └─ LinearRegression (lr_daily_model.pkl)
   (Cùng 3 models như /daily-predict/)

✅ KẾT QUẢ:
   → Bảng 6 dòng (03/04 - 08/04/2026)
   → Mỗi dòng: ngày + RF + XGB + LR + thời tiết dự báo


════════════════════════════════════════════════════════════════════════════

🔑 SỰ KHÁC BIỆT CHÍNH:

┌────────────────────────┬──────────────────────┬──────────────────────────┐
│ Tính năng              │ /daily-predict/      │ /predict/ (Rolling)      │
├────────────────────────┼──────────────────────┼──────────────────────────┤
│ Dữ liệu đầu vào        │ Thực tế (đã xảy ra)  │ Dự báo thời tiết API     │
│ (từ daily_features.csv)│ đến hôm nay          │ + dự đoán trước đó       │
├────────────────────────┼──────────────────────┼──────────────────────────┤
│ Lag features           │ Từ dữ liệu thực tế   │ Mix: thực tế + dự báo    │
│                        │ (chính xác 100%)     │ → ít chính xác hơn       │
├────────────────────────┼──────────────────────┼──────────────────────────┤
│ Dự đoán                │ NGÀY MAI (D+1)       │ 6-7 NGÀY TIẾP THEO       │
│                        │ 1 giá trị            │ (D+2 → D+8)              │
│                        │                      │ 6-7 giá trị              │
├────────────────────────┼──────────────────────┼──────────────────────────┤
│ Độ chính xác           │ Cao                  │ Thấp dần (càng xa = sai) │
│                        │ (dùng dữ liệu thực)  │ (dùng dữ liệu dự báo)    │
├────────────────────────┼──────────────────────┼──────────────────────────┤
│ Thời tiết từ           │ Open-Meteo archive   │ Open-Meteo forecast      │
│                        │ (lịch sử)            │ (dự báo)                 │
└────────────────────────┴──────────────────────┴──────────────────────────┘


════════════════════════════════════════════════════════════════════════════

📍 CHI TIẾT DỮ LIỆU THỜI TIẾT:

1. ARCHIVE API (cho /daily-predict/):
   ├─ URL: https://archive-api.open-meteo.com/v1/archive
   ├─ Phạm vi: 1979 - hôm nay (lịch sử)
   ├─ Chính xác: 100% (dữ liệu thực tế sau)
   └─ Dùng cho: Lag features hôm nay
      (để dự đoán ngày mai)

2. FORECAST API (cho /predict/):
   ├─ URL: https://api.open-meteo.com/v1/forecast
   ├─ Phạm vi: Hôm nay + 7-10 ngày tiếp theo
   ├─ Chính xác: Biến số (dự báo thời tiết)
   └─ Dùng cho: Cập nhật lag features ngày tiếp theo
      (để rolling forecast)

Vị trí Đà Nẵng (Việt Nam):
   Latitude: 16.0678°N
   Longitude: 108.2208°E


════════════════════════════════════════════════════════════════════════════

🎯 TỔNG KẾT:

   daily_features.csv (thực tế: 1979-2026)
   ├─ Dòng cuối cùng (01/04/2026)
   │  └─ /daily-predict/ → Dự đoán ngày mai (D+1)
   │
   └─ + Forecast API (02/04 - 08/04)
      └─ /predict/ → Rolling forecast 6-7 ngày

════════════════════════════════════════════════════════════════════════════
""")
