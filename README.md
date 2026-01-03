# 🌧️ Rainfall Prediction System - Dự Báo Lượng Mưa Đà Nẵng

## 📋 Mục Lục
- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Mô hình Machine Learning](#mô-hình-machine-learning)
- [Kết quả đánh giá](#kết-quả-đánh-giá)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)

---

## 🎯 Giới Thiệu

Hệ thống dự báo lượng mưa hàng tháng tại Đà Nẵng sử dụng **Machine Learning** và **Web Application** để:
- 📊 Phân tích dữ liệu lịch sử lượng mưa (1979-2024)
- 🤖 Xây dựng mô hình dự báo sử dụng Random Forest, LSTM, ARIMA
- 🌐 Cung cấp giao diện web để dự đoán và xem lịch sử
- 📈 Visualize dữ liệu và xu hướng theo mùa

**Dataset**: 45 năm dữ liệu lượng mưa ngày tại Đà Nẵng từ 1979 đến 2024

---

## ✨ Tính Năng

### 1. **Dự Báo Lượng Mưa**
- Nhập năm (1979-2100) và tháng (1-12)
- Dự đoán lượng mưa cho tháng đó
- Hiển thị metrics đánh giá (MAE, RMSE, Accuracy)

### 2. **Phân Tích Dữ Liệu**
- Thống kê chi tiết (mean, median, min, max, std dev)
- Biểu đồ phân phối lượng mưa
- Xu hướng theo mùa (seasonal pattern)
- Heatmap mưa theo năm và tháng

### 3. **Đánh Giá Mô Hình**
- Train/Test split (80/20)
- Metrics: MAE, RMSE, R²
- Feature importance
- Biểu đồ so sánh Actual vs Predicted
- Phân tích phần dư (Residuals)

### 4. **Giao Diện Web**
- Dashboard hiển thị thống kê
- Biểu đồ (Chart.js) cho yearly & monthly data
- Form dự đoán interactif
- Lịch sử dự đoán (cho user đã login)

### 5. **Mô Hình Nâng Cao**
- **Random Forest**: 200 cây, hiệu suất cao
- **LSTM**: Deep learning, bắt được temporal patterns
- **ARIMA**: Time series prediction, phân tích trend & seasonality

---

## 💻 Yêu Cầu Hệ Thống

### Cần cài đặt:
- Python 3.7+
- Django 3.2+
- pip (package manager)

### Thư viện Python:
```
pandas>=1.0.0
numpy>=1.18.0
scikit-learn>=0.24.0
matplotlib>=3.1.0
seaborn>=0.11.0
statsmodels>=0.12.0
tensorflow>=2.4.0 (tùy chọn, cho LSTM)
```

---

## 🚀 Cài Đặt

### 1. Clone hoặc download project
```bash
cd "Du Bao Luong Mua"
```

### 2. Tạo virtual environment (khuyến khích)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 4. Cài đặt packages (nếu requirements.txt không có)
```bash
pip install django pandas numpy scikit-learn matplotlib seaborn statsmodels
# Optional cho LSTM:
pip install tensorflow keras
```

### 5. Tạo database Django
```bash
python manage.py migrate
python manage.py createsuperuser  # Tạo account admin
```

---

## 📖 Sử Dụng

### **1. Chạy Pipeline ML Đầy Đủ** (Từ data → model)

#### Bước 1: Tiền xử lý dữ liệu (chuyển ngày → tháng)
```bash
cd DuBao
python src/preprocess.py
```
📁 Output: `data/monthly_rainfall.csv`

#### Bước 2: Đánh giá và Train mô hình
```bash
python src/evaluate.py
```
📊 Outputs:
- `models/rainfall_model.pkl` - Mô hình Random Forest
- `models/predictions_plot.png` - Biểu đồ Actual vs Predicted
- `models/residuals_plot.png` - Phân tích phần dư

#### Bước 3: Tạo visualizations và EDA
```bash
python src/visualize.py
```
📊 Outputs:
- `models/daily_distribution.png` - Phân phối lượng mưa hàng ngày
- `models/monthly_timeseries.png` - Chuỗi thời gian & mùa
- `models/monthly_heatmap.png` - Heatmap year × month

#### Bước 4: Train các mô hình nâng cao (tùy chọn)
```bash
python src/advanced_models.py
```
🤖 Trains:
- **LSTM** → `models/lstm_model.h5`
- **ARIMA** → `models/arima_model.pkl`
- So sánh 3 mô hình

#### Bước 5: Chạy từ main.py (toàn bộ pipeline)
```bash
python main.py
```

### **2. Chạy Web Application**

```bash
# Từ thư mục gốc của project
python manage.py runserver
```

Mở browser: **http://localhost:8000/**

#### Chức năng trên Web:
- **Trang chủ**: Xem thống kê, biểu đồ
- **Form dự đoán**: Nhập năm/tháng → lấy kết quả dự báo
- **Login**: Đăng nhập để lưu lịch sử dự đoán
- **API Endpoints**:
  - `POST /predict/` - Dự đoán
  - `GET /chart-data/?type=yearly` - Lấy dữ liệu biểu đồ
  - `GET /history/` - Lịch sử dự đoán

---

## 🏗️ Kiến Trúc Hệ Thống

```
Du Bao Luong Mua/
├── DuBao/                          # ML Pipeline
│   ├── main.py                     # Entry point
│   ├── data/
│   │   ├── raw_daily.csv           # Dữ liệu hàng ngày (1979-2024)
│   │   └── monthly_rainfall.csv    # Dữ liệu hàng tháng
│   ├── models/
│   │   ├── rainfall_model.pkl      # Mô hình Random Forest
│   │   ├── lstm_model.h5           # Mô hình LSTM
│   │   ├── arima_model.pkl         # Mô hình ARIMA
│   │   ├── predictions_plot.png    # Biểu đồ đánh giá
│   │   ├── residuals_plot.png      # Phân tích phần dư
│   │   ├── daily_distribution.png  # Phân phối lượng mưa
│   │   ├── monthly_timeseries.png  # Chuỗi thời gian
│   │   └── monthly_heatmap.png     # Heatmap
│   └── src/
│       ├── preprocess.py           # Tiền xử lý dữ liệu
│       ├── train_model.py          # Train Random Forest
│       ├── predict.py              # Dự đoán
│       ├── evaluate.py             # Đánh giá mô hình
│       ├── visualize.py            # EDA & Visualization
│       ├── advanced_models.py      # LSTM & ARIMA
│       └── utils.py                # Utilities
│
├── predictor/                       # Django App
│   ├── models.py                   # RainfallPrediction model
│   ├── views.py                    # Views & APIs
│   ├── urls.py                     # URL routing
│   ├── admin.py                    # Django admin
│   └── templates/
│       ├── predictor/
│       │   └── index.html          # Dashboard
│       └── registration/
│           └── login.html          # Login page
│
├── rainfall_project/                # Django config
│   ├── settings.py                 # Cài đặt Django
│   ├── urls.py                     # URL chính
│   └── wsgi.py                     # WSGI config
│
├── manage.py                        # Django management
└── db.sqlite3                       # Database
```

---

## 🤖 Mô Hình Machine Learning

### **1. Random Forest Regressor**
```python
• Số cây: 200
• Điểm mạnh: Nhanh, không cần scaling, xử lý nonlinear
• Sử dụng: Features [year, month] → dự đoán rainfall
```

**Kết quả:**
```
Training MAE:  X.XX mm
Testing MAE:   Y.YY mm
Testing RMSE:  Z.ZZ mm
Testing R²:    A.AA
```

### **2. LSTM (Deep Learning)**
```python
• 2 LSTM layers (50 units each)
• Dropout: 0.2 (prevent overfitting)
• Lookback window: 12 tháng
• Điểm mạnh: Bắt được temporal dependencies
```

**Kết quả:**
```
Training MAE:  X.XX mm
Testing MAE:   Y.YY mm
Testing R²:    A.AA
```

### **3. ARIMA (Time Series)**
```python
• Order: (p, d, q) - Auto-regressive Integrated Moving Average
• p: AR order (từ quá khứ)
• d: Degree of differencing
• q: MA order (từ lỗi quá khứ)
• Điểm mạnh: Phân tích trend & seasonality rõ ràng
```

---

## 📊 Kết Quả Đánh Giá

### **Dataset Statistics**
```
📈 Dữ liệu hàng ngày (1979-2024):
   • Tổng records: ~12,787
   • Trung bình: X.XX mm/ngày
   • Cao nhất: XXX.XX mm/ngày
   • Thấp nhất: 0 mm

📈 Dữ liệu hàng tháng:
   • Tổng tháng: 546 (45 năm × 12)
   • Trung bình: XXX.XX mm/tháng
   • Mưa cao nhất tháng: XXX.XX mm
```

### **Model Comparison**
| Mô hình | MAE (mm) | RMSE (mm) | R² | Tốc độ |
|---------|----------|-----------|-----|--------|
| Random Forest | Y.YY | Z.ZZ | A.AA | ⚡⚡⚡ |
| LSTM | Y.YY | Z.ZZ | A.AA | ⚡⚡ |
| ARIMA | Y.YY | Z.ZZ | A.AA | ⚡⚡⚡ |

### **Metrics Giải Thích**
- **MAE (Mean Absolute Error)**: Sai số trung bình (đơn vị: mm)
- **RMSE (Root Mean Squared Error)**: Căn bậc hai sai số (mm)
- **R² (Coefficient of Determination)**: Tỷ lệ phương sai được giải thích
  - R² > 0.8: Excellent model ✅
  - R² > 0.6: Good model ✅
  - R² > 0.4: Moderate model ⚠️
  - R² < 0.4: Weak model ❌

---

## 🛠️ Công Nghệ Sử Dụng

### **Backend**
| Công nghệ | Phiên bản | Mục đích |
|-----------|----------|---------|
| Django | 3.2+ | Web framework, API endpoints |
| Python | 3.7+ | Ngôn ngữ lập trình |
| SQLite | Latest | Database |

### **Data Science**
| Thư viện | Phiên bản | Mục đích |
|---------|----------|---------|
| Pandas | 1.0+ | Data manipulation |
| NumPy | 1.18+ | Numerical computing |
| Scikit-learn | 0.24+ | Machine Learning models |
| TensorFlow/Keras | 2.4+ | Deep Learning (LSTM) |
| Statsmodels | 0.12+ | Time Series (ARIMA) |
| Matplotlib | 3.1+ | Visualization |
| Seaborn | 0.11+ | Statistical visualization |

### **Frontend**
| Công nghệ | Mục đích |
|-----------|---------|
| HTML5 | Cấu trúc trang |
| CSS3 | Styling |
| JavaScript | Interactivity |
| Chart.js | Vẽ biểu đồ |
| Bootstrap | Responsive design |

---

## 📝 Ví Dụ Sử Dụng

### **Python CLI**
```python
from src.predict import predict_rainfall

# Dự đoán lượng mưa tháng 10/2025
prediction = predict_rainfall("models/rainfall_model.pkl", 2025, 10)
print(f"Dự đoán: {prediction:.2f} mm")
```

### **Web API**
```javascript
// Dự đoán từ JavaScript
fetch('/predict/', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({year: 2025, month: 10})
})
.then(r => r.json())
.then(data => console.log(`Dự đoán: ${data.rainfall} mm`))
```

---

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'tensorflow'`
**Giải pháp**: LSTM là tùy chọn, không bắt buộc. Cài đặt nếu cần:
```bash
pip install tensorflow
```

### Error: `No such file or directory: 'data/monthly_rainfall.csv'`
**Giải pháp**: Chạy `preprocess.py` trước:
```bash
python src/preprocess.py
```

### Error: Database locked
**Giải pháp**: Xóa `db.sqlite3` và chạy lại:
```bash
python manage.py migrate
```

---

## 👥 Tác Giả

**Bạn** - Sinh viên Bách Khoa

---

## 📄 License

MIT License - Sử dụng tự do cho mục đích học tập

---

## 📞 Support

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue hoặc liên hệ.

---

## 🎓 Đối Tượng

**Đồ án tốt nghiệp - Bách Khoa**

Dự báo lượng mưa hàng tháng tại Đà Nẵng sử dụng Machine Learning
