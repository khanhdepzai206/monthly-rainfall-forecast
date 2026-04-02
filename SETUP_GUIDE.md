# 🚀 HƯỚNG DẪN CHẠY & TRAIN DỰ ÁN DỰ BÁO LƯỢNG MƯA

## 📋 CÁC CÂU LỆNH CẦN THIẾT

### 1️⃣ SETUP ENVIRONMENT (Lần đầu tiên)

```bash
# 1. Cài đặt dependencies Python
cd "d:\Du Bao Luong Mua"
pip install -r requirements.txt

# 2. Tạo database Django
python manage.py makemigrations
python manage.py migrate

# 3. Tạo superuser (tùy chọn)
python manage.py createsuperuser
```

### 2️⃣ TRAIN MODELS (Nếu chưa có models hoặc muốn train lại)

```bash
# Di chuyển vào thư mục ML
cd "d:\Du Bao Luong Mua\DuBao\src"

# Train 3 models chính (RF, LR, XGB)
python train_daily_models.py

# Hoặc train từng model riêng lẻ:
python train_daily_model.py  # RandomForest
python train_xgb_simple.py   # XGBoost
python train_model.py        # LinearRegression
```

### 3️⃣ CHẠY DJANGO SERVER

```bash
# Từ thư mục gốc dự án
cd "d:\Du Bao Luong Mua"
python manage.py runserver
```

### 4️⃣ TRUY CẬP WEB INTERFACE

Sau khi server chạy, mở browser và truy cập:

- **Trang chủ**: http://127.0.0.1:8000/
- **Dự đoán ngày mai**: http://127.0.0.1:8000/daily-predict/
- **Nhập actual rainfall**: http://127.0.0.1:8000/actual-input/
- **Xem metrics**: http://127.0.0.1:8000/metrics/

## 🔄 WORKFLOW HÀNG NGÀY

### Buổi Sáng: Dự Đoán
```bash
# 1. Mở browser
start http://127.0.0.1:8000/daily-predict/

# 2. Nhấn nút "Dự Đoán Ngày Mai"
# 3. Xem kết quả 3 models
```

### Buổi Chiều: Cập Nhật Actual
```bash
# 1. Mở browser
start http://127.0.0.1:8000/actual-input/

# 2. Nhập ngày và lượng mưa thực tế
# 3. Submit (tự động tính error và retrain nếu cần)
```

## 🛠️ CÂU LỆNH BẢO TRÌ

### Kiểm tra models có tồn tại không
```bash
cd "d:\Du Bao Luong Mua\DuBao\models"
dir rf_daily_model.pkl lr_daily_model.pkl xgb_daily_model.pkl
```

### Test ML pipeline
```bash
cd "d:\Du Bao Luong Mua\DuBao\src"
python run_pipeline.py
```

### Xem logs database
```bash
cd "d:\Du Bao Luong Mua"
python manage.py shell
# Trong shell:
from predictor.models import DailyPrediction, ActualRainfall
print("Predictions:", DailyPrediction.objects.count())
print("Actuals:", ActualRainfall.objects.count())
```

### Manual retrain (nếu cần)
```bash
cd "d:\Du Bao Luong Mua\DuBao\src"
python run_pipeline.py retrain
```

## ⚠️ LƯU Ý QUAN TRỌNG

### Khi nào CẦN train lại:
- ❌ **Lần đầu setup**: Cần train models
- ❌ **Models bị xóa**: Cần train lại
- ❌ **Thêm data mới nhiều**: Nên retrain để cải thiện
- ✅ **Chạy hàng ngày**: KHÔNG cần train, chỉ dùng models có sẵn
- ✅ **Auto retrain**: Tự động khi error > 20%

### Khi nào KHÔNG cần train:
- ✅ **Models đã tồn tại** trong `DuBao/models/`
- ✅ **Chỉ muốn predict**: Dùng `run_pipeline.py`
- ✅ **Chạy web interface**: Models tự động load

## 🚨 TROUBLESHOOTING

### Lỗi "No module named..."
```bash
pip install -r requirements.txt
```

### Lỗi "Models not found"
```bash
cd "d:\Du Bao Luong Mua\DuBao\src"
python train_daily_models.py
```

### Lỗi database
```bash
cd "d:\Du Bao Luong Mua"
python manage.py makemigrations
python manage.py migrate
```

### Server không start
```bash
# Kiểm tra port 8000 có bị chiếm không
netstat -ano | findstr :8000
# Kill process nếu cần
taskkill /PID <PID> /F
```

## 📁 CẤU TRÚC THỤ MỤC

```
d:\Du Bao Luong Mua\
├── manage.py                    # Django entry point
├── requirements.txt             # Python dependencies
├── db.sqlite3                   # Database
├── predictor/                   # Django app
│   ├── models.py               # Database models
│   ├── views.py                # Web views
│   ├── forms.py                # Forms
│   └── urls.py                 # URL routing
├── templates/                   # HTML templates
├── DuBao/                       # ML pipeline
│   ├── models/                 # Trained models (.pkl)
│   └── src/                    # ML source code
└── README.md                   # Documentation
```

## 🎯 QUICK START (Cho người mới)

```bash
# 1. Setup (chỉ 1 lần)
cd "d:\Du Bao Luong Mua"
pip install -r requirements.txt
python manage.py migrate

# 2. Train models (chỉ 1 lần)
cd "DuBao\src"
python train_daily_models.py

# 3. Chạy server
cd "../.."
python manage.py runserver

# 4. Mở browser
start http://127.0.0.1:8000/daily-predict/
```

**🎉 Sẵn sàng sử dụng!**</content>
<parameter name="filePath">d:\Du Bao Luong Mua\SETUP_GUIDE.md