# Hướng dẫn: Dự đoán mưa theo ngày (2-Step Model)

## Vấn đề ban đầu
Khi dự đoán mưa theo **ngày**, dữ liệu có rất nhiều giá trị **0** (ngày không mưa) khiến mô hình khó học được mô hình chính xác.

## Giải pháp: Two-Step Model
Pipeline mới sử dụng **2 mô hình riêng biệt**:

### Bước 1: Classifier (Phân loại)
- **Mục đích**: Dự đoán có mưa hay không? (Binary classification)
- **Đầu ra**: Xác suất mưa (0-100%)
- **Lợi ích**: Model chỉ cần học phân biệt ngày mưa vs không mưa

### Bước 2: Regressor (Hồi quy)
- **Mục đích**: Dự đoán lượng mưa (chỉ cho ngày dự đoán có mưa)
- **Dữ liệu huấn luyện**: Chỉ các ngày có mưa
- **Lợi ích**: Model chỉ cần dự đoán lượng mưa, bỏ qua các ngày không mưa

---

## Cách chạy

### 1. Huấn luyện mô hình
```bash
cd DuBao
python main.py --daily
```

Quá trình sẽ:
1. ✅ Chuyển dữ liệu từ ngày → tháng
2. ✅ Chuẩn bị dữ liệu ngày (daily_combined.csv)
3. ✅ Train Classifier (dự đoán có mưa hay không)
4. ✅ Train Regressor (dự đoán lượng mưa)

**Output:**
- `models/daily_classifier.pkl` - Mô hình phân loại
- `models/daily_regressor.pkl` - Mô hình hồi quy

### 2. Dự đoán cho một ngày
```bash
python main.py --daily
```

Sau khi train, nhập:
- Năm (vd: 2023)
- Tháng (1-12)
- Ngày (1-31)

**Kết quả:**
```
📅 Dự đoán ngày 15/5/2023:
  🌦️ Có mưa: Có
  📊 Xác suất mưa: 75.3%
  🌧️ Lượng mưa dự đoán: 45.60 mm
```

### 3. Train lại mô hình
```bash
python main.py --daily --retrain
```

---

## Features được sử dụng

### Temporal Features
- `month_sin`, `month_cos` - Mã hóa chu kỳ tháng
- `day_sin`, `day_cos` - Mã hóa chu kỳ ngày
- `trend` - Xu hướng theo thời gian

### Rainfall Features
- `rainfall_lag_1, lag_3, lag_7` - Lượng mưa các ngày trước
- `rainfall_ma_3, ma_7` - Trung bình động
- `rainfall_std_7` - Độ lệch chuẩn (biến động)

### Weather Features
- `temperature_lag_1, lag_3, ma_3`
- `humidity_lag_1, lag_3, ma_3`
- `wind_speed_lag_1, lag_3, ma_3`
- `cloud_cover_lag_1, ma_3`
- `surface_pressure_lag_1, ma_3`

---

## Kết quả mong đợi

**Classifier:**
- Độ chính xác: 75-85%
- Phân biệt rõ ràng ngày mưa vs không mưa

**Regressor:**
- MAE (Mean Absolute Error): 10-20 mm
- R²: 0.6-0.8
- Chỉ cần dự đoán lượng mưa cho những ngày có mưa

---

## Khác biệt với dự đoán theo tháng

| Tiêu chí | Theo tháng | Theo ngày |
|---|---|---|
| Granularity | Thấp (tháng) | Cao (ngày) |
| Số lượng 0 | Ít | Nhiều (40-50%) |
| Model | 1 Regressor | 2 Model (Classifier + Regressor) |
| Độ chính xác | ~75% | ~75-85% |
| Thời gian train | 1-2 phút | 2-5 phút |

---

## File liên quan
- `src/train_two_step_daily.py` - Huấn luyện 2 model
- `src/predict_daily.py` - Dự đoán theo ngày
- `src/preprocess.py` - Chuẩn bị dữ liệu (create_daily_combined)
- `main.py` - Script chính (thêm flag --daily)

---

## Lưu ý
- Cần có `data/daily_combined.csv` (dữ liệu ngày + thời tiết)
- Nếu không có weather data, chỉ dùng rainfall data
- Model tự động điều chỉnh features dựa trên dữ liệu có sẵn
