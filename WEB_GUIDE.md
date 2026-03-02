# 🚀 Hướng dẫn: Từ Train Mô hình Đến Web

## 📋 Quy trình tổng quan

```
┌─────────────────────┐
│  Train Mô hình      │ (Terminal 1)
│ python main.py      │
│   --compare         │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Kiểm tra Mô hình   │
│ (Files lưu tại      │
│  models/*.pkl)      │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Chạy Django Web    │
│ python manage.py    │
│   runserver         │ (Terminal 2)
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Truy cập Web       │
│ http://localhost:   │
│   8000/             │
└─────────────────────┘
```

---

## 🔧 Bước 1: Train Mô hình (5-10 phút)

### Terminal 1: Train mô hình

```bash
# Di chuyển vào thư mục dự án
cd "d:\Du Bao Luong Mua\DuBao"

# So sánh 3 mô hình (GradientBoosting, RandomForest, XGBoost)
python main.py --compare
```

**Kết quả:**
```
═════════════════════════════════════════
🔬 CHẾ ĐỘ SO SÁNH MÔ HÌNH
═════════════════════════════════════════

📊 SO SÁNH CLASSIFIER MODELS
╔═════════════════╦═══════════╦═══════════╦═════════╦═════════╦═══════════╗
║ Model           ║ Accuracy  ║ Precision ║ Recall  ║ F1-Score║ Train (s) ║
╠═════════════════╬═══════════╬═══════════╬═════════╬═════════╬═══════════╣
║ GradientBoosting║   0.8234  ║   0.7543  ║  0.8901 ║  0.8167 ║   1.23    ║
║ RandomForest    ║   0.8121  ║   0.7401  ║  0.8723 ║  0.8032 ║   2.45    ║
║ XGBoost         ║   0.8456  ║   0.7834  ║  0.9012 ║  0.8389 ║   1.89    ║
╚═════════════════╩═══════════╩═══════════╩═════════╩═════════╩═══════════╝

🏆 Classifier tốt nhất: XGBoost (F1: 0.8389)

📊 SO SÁNH REGRESSOR MODELS
╔═════════════════╦═══════════╦═══════════╦═══════╦═════════╦═══════════╗
║ Model           ║ MAE (mm)  ║ RMSE (mm) ║  R²   ║ MAPE (%)║ Train (s) ║
╠═════════════════╬═══════════╬═══════════╬═══════╬═════════╬═══════════╣
║ GradientBoosting║   12.34   ║   18.92   ║0.7234 ║ 15.23   ║   2.12    ║
║ RandomForest    ║   13.45   ║   19.87   ║0.7012 ║ 16.34   ║   3.45    ║
║ XGBoost         ║   11.23   ║   17.45   ║0.7456 ║ 14.12   ║   2.34    ║
╚═════════════════╩═══════════╩═══════════╩═══════╩═════════╩═══════════╝

🏆 Regressor tốt nhất: XGBoost (R²: 0.7456)

💾 Lưu kết quả so sánh...
✅ Đã lưu comparison_results.pkl
```

**Kiểm tra file được tạo:**
```bash
# Xem các file mô hình
dir models\

# Output:
daily_classifier_gradientboosting.pkl
daily_classifier_randomforest.pkl
daily_classifier_xgboost.pkl
daily_regressor_gradientboosting.pkl
daily_regressor_randomforest.pkl
daily_regressor_xgboost.pkl
comparison_results.pkl                  ← File này rất quan trọng!
```

---

## 🌐 Bước 2: Chạy Django Web Server (Terminal 2)

### Terminal 2: Chạy web server

```bash
# Di chuyển vào thư mục chính
cd "d:\Du Bao Luong Mua"

# Khởi động Django development server
python manage.py runserver

# Hoặc chỉ định port
python manage.py runserver 8000
```

**Kết quả:**
```
Watching for file changes with StatReloader
Performing system checks...

System check identified no issues (0 silenced).
January 30, 2026 - 10:30:45
Django version 4.0.x, using settings 'rainfall_project.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

---

## 🌍 Bước 3: Truy cập Web

### Mở trình duyệt và dùng các trang:

#### **1. Trang chính**
```
http://localhost:8000/
```
- Hiển thị tổng quan về dự đoán

#### **2. Dự đoán Theo Ngày** (MỚI! 🆕)
```
http://localhost:8000/predictor/predict-daily/
```
- **Form:** Nhập năm, tháng, ngày
- **Kết quả:** 
  - ✅ Có mưa hay không?
  - 📊 Xác suất mưa (%)
  - 💧 Lượng mưa (mm)
  - 🎯 Tên mô hình được dùng

#### **3. Dự đoán Khoảng Ngày** 
```
http://localhost:8000/predictor/predict-daily/
```
- Nhập năm, tháng, ngày bắt đầu, số ngày
- Hiển thị bảng dự đoán 10 ngày liên tiếp

#### **4. Các trang khác**
```
http://localhost:8000/predict/              # Dự đoán theo tháng
http://localhost:8000/compare/           # So sánh mô hình
http://localhost:8000/history/              # Lịch sử dự đoán (legacy)
```

---

## 🔌 API Endpoints (JSON)

### 1. Dự đoán một ngày

**Endpoint:** `POST /predictor/api/predict/`

**Request:**
```json
{
    "year": 2023,
    "month": 5,
    "day": 15
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "year": 2023,
        "month": 5,
        "day": 15,
        "date": "2023-05-15",
        "has_rain": true,
        "rain_probability": 0.753,
        "predicted_rainfall": 45.60,
        "classifier_model": "XGBoost",
        "regressor_model": "XGBoost"
    },
    "message": "Dự đoán thành công"
}
```

**Test với curl:**
```bash
curl -X POST http://localhost:8000/predictor/api/predict/ ^
  -H "Content-Type: application/json" ^
  -d "{\"year\": 2023, \"month\": 5, \"day\": 15}"
```

---

### 2. Dự đoán khoảng ngày

**Endpoint:** `GET /predictor/api/predict-range/`

**Query params:**
```
year=2023&month=5&start_day=1&num_days=10
```

**URL:**
```
http://localhost:8000/predictor/api/predict-range/?year=2023&month=5&start_day=1&num_days=10
```

**Response:**
```json
{
    "success": true,
    "data": [
        {
            "year": 2023,
            "month": 5,
            "day": 1,
            "date": "2023-05-01",
            "has_rain": false,
            "rain_probability": 0.234,
            "predicted_rainfall": 0.0,
            "classifier_model": "XGBoost",
            "regressor_model": "XGBoost"
        },
        ...
    ],
    "count": 10,
    "message": "Dự đoán 10 ngày thành công"
}
```

---

### 3. Lấy thông tin mô hình

**Endpoint:** `GET /predictor/api/model-info/`

**URL:**
```
http://localhost:8000/predictor/api/model-info/
```

**Response:**
```json
{
    "success": true,
    "best_classifier": "XGBoost",
    "best_regressor": "XGBoost",
    "available_classifiers": ["GradientBoosting", "RandomForest", "XGBoost"],
    "available_regressors": ["GradientBoosting", "RandomForest", "XGBoost"],
    "message": "Thông tin mô hình"
}
```

---

## 📱 Ví dụ sử dụng (Postman hoặc Python)

### Python:
```python
import requests

# Dự đoán một ngày
response = requests.post('http://localhost:8000/predictor/api/predict/', 
    json={'year': 2023, 'month': 5, 'day': 15})

result = response.json()
print(f"Ngày 15/5/2023:")
print(f"  Có mưa: {'Có' if result['data']['has_rain'] else 'Không'}")
print(f"  Xác suất: {result['data']['rain_probability']*100:.1f}%")
print(f"  Lượng mưa: {result['data']['predicted_rainfall']:.2f} mm")
```

### cURL (Windows):
```bash
# Dự đoán một ngày
curl -X POST http://localhost:8000/predictor/api/predict/ ^
  -H "Content-Type: application/json" ^
  -d "{\"year\": 2023, \"month\": 5, \"day\": 15}"

# Lấy thông tin mô hình
curl http://localhost:8000/predictor/api/model-info/

# Dự đoán 10 ngày
curl "http://localhost:8000/predictor/api/predict-range/?year=2023&month=5&start_day=1&num_days=10"
```

---

## 🛑 Tắt Server

Nhấn `CTRL + C` trong terminal:
```
^CKeyboardInterrupt
```

---

## ✅ Checklist

- [ ] Train mô hình: `python main.py --compare`
- [ ] Kiểm tra files: `models/daily_classifier_*.pkl` tồn tại
- [ ] Chạy server: `python manage.py runserver`
- [ ] Truy cập web: `http://localhost:8000/predictor/`
- [ ] Test API: `POST /predictor/api/predict/`

---

## 🐛 Troubleshooting

### Lỗi 1: "ModuleNotFoundError: No module named 'DuBao'"
```bash
# Giải pháp: Cài lại packages
pip install -r requirements.txt
```

### Lỗi 2: "FileNotFoundError: comparison_results.pkl"
```bash
# Giải pháp: Chạy train mô hình trước
cd DuBao
python main.py --compare
```

### Lỗi 3: Port 8000 đang bận
```bash
# Giải pháp: Dùng port khác
python manage.py runserver 8001
```

### Lỗi 4: ModuleNotFoundError XGBoost
```bash
# Giải pháp: Cài XGBoost
pip install xgboost
```

---

## 📊 Dữ liệu đầu vào & đầu ra

### Input Features (từ daily_combined.csv):
- Rainfall (ngày trước)
- Temperature, Humidity, Wind Speed
- Cloud Cover, Surface Pressure
- Lag features (1, 3, 7 ngày trước)
- Moving averages (3, 7 ngày)
- Cyclical features (tháng, ngày)

### Output Predictions:
- `has_rain`: Boolean (có/không mưa)
- `rain_probability`: Float 0-1 (xác suất mưa)
- `predicted_rainfall`: Float mm (lượng mưa dự đoán)

---

## 🎯 Kết quả mong đợi

| Metric | Classifier | Regressor |
|--------|-----------|-----------|
| Accuracy/R² | ~80-84% | ~0.72-0.75 |
| Precision/MAE | ~75-80% | 11-13 mm |
| Recall/RMSE | ~88-90% | 17-19 mm |
| F1/MAPE | ~0.82 | 14-15% |

---

**Bạn sẵn sàng chạy chưa? 🚀**
