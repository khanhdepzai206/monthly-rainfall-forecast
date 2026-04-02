# 🌧️ Hệ Thống Dự Báo Lượng Mưa Ngày Mai - Django Web Interface

Hệ thống web Django hoàn chỉnh để dự đoán lượng mưa ngày mai sử dụng 3 mô hình Machine Learning và tự động cải thiện độ chính xác.

## ✨ Tính Năng

### 🧠 Dự Đoán Thông Minh
- **3 Mô Hình ML**: RandomForest, XGBoost, LinearRegression
- **Dự Đoán Ngày Mai**: Nhấn nút để predict lượng mưa ngày mai
- **Hiển Thị Kết Quả**: 3 giá trị dự đoán + mô hình tốt nhất
- **Tự Động Lưu**: Kết quả lưu vào database Django

### 📊 Cải Thiện Độ Chính Xác
- **Nhập Actual Rainfall**: Form nhập lượng mưa thực tế
- **Tự Động Tính Sai Số**: So sánh prediction vs actual
- **Auto Retrain**: Retrain models khi sai số > 20%
- **Theo Dõi Lịch Sử**: Xem tất cả actual data và errors

### 🎨 Giao Diện Đẹp
- **Responsive Design**: Bootstrap 5, mobile-friendly
- **Real-time Updates**: AJAX, loading states
- **Visual Feedback**: Icons, colors, success messages
- **Navigation**: Menu dễ sử dụng

## 🚀 Cài Đặt & Chạy

### 1. Chuẩn Bị Models
```bash
cd "d:\Du Bao Luong Mua\DuBao\src"
python daily_ml_system.py setup
```

### 2. Chạy Server Django
```bash
cd "d:\Du Bao Luong Mua"
python manage.py runserver
```

### 3. Truy Cập
- **Trang Dự Báo**: http://127.0.0.1:8000/daily-predict
- **Trang Nhập Actual**: http://127.0.0.1:8000/actual-input

## 📋 Workflow Sử Dụng

### Buổi Sáng: Dự Đoán
1. Truy cập `/daily-predict`
2. Nhấn nút "Dự Đoán Ngày Mai"
3. Xem kết quả 3 mô hình
4. Hệ thống tự động lưu prediction

### Buổi Chiều: Cập Nhật Actual
1. Truy cập `/actual-input`
2. Nhập ngày và lượng mưa thực tế
3. Submit form
4. Hệ thống:
   - Tính sai số cho từng mô hình
   - Kiểm tra ngưỡng retrain (20%)
   - Auto retrain nếu cần
   - Cập nhật database

## 🗂️ Cấu Trúc Code

### Models (models.py)
```python
class DailyPrediction(models.Model):
    date = models.DateField(unique=True)
    rf_pred = models.FloatField()
    lr_pred = models.FloatField()
    xgb_pred = models.FloatField()
    best_model = models.CharField(max_length=10, null=True)

class ActualRainfall(models.Model):
    date = models.DateField(unique=True)
    actual_rainfall = models.FloatField()
    prediction = models.OneToOneField(DailyPrediction)
    rf_error = models.FloatField(null=True)
    lr_error = models.FloatField(null=True)
    xgb_error = models.FloatField(null=True)
    retrained = models.BooleanField(default=False)
```

### Views (views.py)
```python
def daily_predict(request):
    """Dự đoán ngày mai bằng 3 models"""
    if request.method == 'POST':
        pred_rf, pred_lr, pred_xgb = get_daily_predictions()
        # Lưu vào DB và hiển thị

def actual_input(request):
    """Nhập actual rainfall"""
    if request.method == 'POST':
        # Lưu actual, tính error, auto retrain
```

### URLs (urls.py)
```python
path('daily-predict/', views.daily_predict, name='daily_predict'),
path('actual-input/', views.actual_input, name='actual_input'),
```

### Templates
- `daily_predict.html`: Trang dự đoán với nút và kết quả
- `actual_input.html`: Form nhập actual và lịch sử

## 🔧 Tích Hợp Với run_pipeline.py

### Function get_daily_predictions()
```python
def get_daily_predictions():
    """Load 3 models và predict cho ngày mai"""
    # Load models từ models/rf_daily_model.pkl, etc.
    # Load latest features từ data/daily_features.csv
    # Return pred_rf, pred_lr, pred_xgb
```

### Function retrain_models()
```python
def retrain_models():
    """Retrain tất cả models với data mới"""
    from train_daily_models import train_daily_models
    train_daily_models()
```

## 📊 Cơ Chế Auto Retrain

### Logic Retrain
```python
def check_retrain_threshold(self, threshold_percent=20):
    """Kiểm tra sai số > 20% so với actual rainfall"""
    for error in [rf_error, lr_error, xgb_error]:
        if error and (error / actual_rainfall * 100) > threshold_percent:
            return True
    return False
```

### Khi Retrain
- Gọi `retrain_models()` từ run_pipeline.py
- Models được train lại với toàn bộ data
- Files model.pkl được ghi đè
- Flag `retrained = True` trong ActualRainfall

## 🎯 Ví Dụ Sử Dụng

### Dự Đoán Ngày Mai
```
User nhấn "Dự Đoán Ngày Mai"
→ Gọi get_daily_predictions()
→ Hiển thị: RF: 3.45mm, XGB: 2.89mm, LR: 1.23mm
→ Lưu vào DailyPrediction model
```

### Nhập Actual Rainfall
```
User nhập: Ngày 15/1/2024, Rainfall 2.1mm
→ Tính errors: RF:1.35, XGB:0.79, LR:0.87
→ Sai số < 20%, không retrain
→ Lưu ActualRainfall với errors
```

### Auto Retrain Trigger
```
User nhập: Ngày 16/1/2024, Rainfall 10.0mm
Prediction: RF:2.0mm, XGB:1.8mm, LR:2.2mm
→ Errors: RF:8.0, XGB:8.2, LR:7.8
→ Sai số > 80%, trigger retrain
→ Gọi retrain_models()
```

## 🔗 API Endpoints

- `GET/POST /daily-predict/`: Dự đoán ngày mai
- `GET/POST /actual-input/`: Nhập actual rainfall
- Database models: DailyPrediction, ActualRainfall

## 📈 Monitoring & Logs

- **Database**: Lưu tất cả predictions và actuals
- **Error Tracking**: Sai số từng mô hình được lưu
- **Retrain History**: Flag retrained trong ActualRainfall
- **Best Model**: Hiển thị mô hình có error thấp nhất

Hệ thống hoàn toàn tự động và sẽ cải thiện độ chính xác theo thời gian khi có nhiều dữ liệu actual! 🚀