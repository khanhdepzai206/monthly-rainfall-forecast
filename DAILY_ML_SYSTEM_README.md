# 🌧️ Hệ Thống Dự Báo Lượng Mưa Hàng Ngày

Hệ thống Machine Learning hoàn chỉnh tự động dự đoán lượng mưa ngày mai sử dụng 3 mô hình: RandomForest, XGBoost, LinearRegression.

## ✨ Tính Năng

- **🔮 Dự đoán hàng ngày**: Tự động predict lượng mưa ngày mai
- **📊 3 Mô Hình ML**: RandomForest, XGBoost, LinearRegression
- **🔄 Tự động retrain**: Retrain khi sai số > 2.0mm (7 ngày trung bình)
- **📈 Theo dõi hiệu suất**: Log và đánh giá model performance
- **🎯 Model tốt nhất**: Tự động xác định model có error thấp nhất- **🌐 Web Interface**: Cập nhật actual rainfall qua giao diện web

## 📋 Yêu Cầu

- Python 3.7+
- Libraries: pandas, numpy, scikit-learn, xgboost
- Django (cho web interface)
## 📋 Yêu Cầu

- Python 3.7+
- Libraries: pandas, numpy, scikit-learn, xgboost

## 🚀 Cách Sử Dụng

### 1. Setup Ban Đầu (Một lần)

```bash
cd "d:\Du Bao Luong Mua\DuBao\src"
python daily_ml_system.py setup
```

### 2. Dự Đoán Hàng Ngày

```bash
# Mỗi sáng, chạy prediction cho ngày hôm đó
python daily_ml_system.py predict
```

### 3. Cập Nhật Actual (Mỗi tối)

```bash
# Sau khi có dữ liệu thực tế
python daily_ml_system.py update --date 2024-01-15 --rainfall 5.2
```

### 4. Kiểm Tra Trạng Thái

```bash
python daily_ml_system.py status
```

## 📁 Cấu Trúc File

```
DuBao/
├── src/
│   ├── daily_ml_system.py      # Script chính
│   ├── prepare_daily_data.py   # Chuẩn bị dữ liệu
│   ├── train_daily_models.py   # Train models
│   ├── predict_daily_new.py    # Dự đoán
│   ├── update_actual.py        # Cập nhật actual
│   ├── auto_retrain.py         # Auto retrain
│   └── evaluate_daily_models.py # Đánh giá
├── data/
│   ├── raw_daily.csv          # Dữ liệu gốc
│   ├── weather_daily.csv      # Dữ liệu thời tiết
│   ├── daily_features.csv     # Features đã xử lý
│   └── prediction_log.csv     # Log predictions
└── models/
    ├── rf_daily_model.pkl     # RandomForest model
    ├── xgb_daily_model.pkl    # XGBoost model
    ├── lr_daily_model.pkl     # LinearRegression model
    ├── daily_model_metrics.csv # Metrics ban đầu
    └── model_evaluation.csv   # Đánh giá hiện tại
```

## 🔧 Chi Tiết Kỹ Thuật

### Features
- **Lag features**: Nhiệt độ, độ ẩm, gió, mây, áp suất của 7 ngày trước
- **Target**: Lượng mưa ngày mai (mm)

### Models
- **RandomForest**: 100 trees, tốt cho non-linear patterns
- **XGBoost**: Gradient boosting, high performance
- **LinearRegression**: Baseline model

### Auto Retrain
- **Trigger**: Sai số trung bình 7 ngày > 2.0mm
- **Action**: Retrain model đó với dữ liệu mới nhất
- **Save**: Model mới thay thế model cũ

### Log Format
```csv
date,rf_pred,xgb_pred,lr_pred,actual,error_rf,error_xgb,error_lr,predicted_at
2024-01-15,3.45,2.89,1.23,2.1,1.35,0.79,0.87,2024-01-15 08:00:00
```

## 📊 Workflow Hàng Ngày

```
Sáng 8:00: Chạy predict → Lưu prediction log
Tối 20:00: Nhập actual rainfall → Tính error → Auto retrain nếu cần → Đánh giá
```

## 🎯 Ví Dụ Sử Dụng

```bash
# Setup
python daily_ml_system.py setup

# Predict ngày mai
python daily_ml_system.py predict

# Cập nhật actual (giả sử mưa 3.5mm ngày 15/1)
python daily_ml_system.py update --date 2024-01-15 --rainfall 3.5

# Xem status
python daily_ml_system.py status
```

## 📈 Monitoring

- **prediction_log.csv**: Theo dõi tất cả predictions và errors
- **model_evaluation.csv**: Performance của từng model
- **Status command**: Quick overview

Hệ thống hoàn toàn tự động và có thể chạy hàng ngày để cải thiện độ chính xác theo thời gian! 🚀