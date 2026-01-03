# 🚀 HƯỚNG DẪN CÀI ĐẶT & CHẠY TOÀN BỘ DỰ ÁN

## 📋 Các Bước Cơ Bản

### 1️⃣ **Cài Đặt Python & Pip**

**Windows:**
- Download Python từ https://www.python.org/downloads/
- Cài đặt, **TICK** "Add Python to PATH"
- Verify: Mở CMD gõ `python --version`

**Linux/Mac:**
```bash
sudo apt-get install python3 python3-pip  # Linux
brew install python3                       # Mac
```

---

### 2️⃣ **Setup Project**

```bash
# Di chuyển vào thư mục dự án
cd "D:\Du Bao Luong Mua"

# Tạo virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài thư viện (copypaste cả dòng)
pip install -r requirements.txt
```

**Lưu ý Windows**: Nếu bị lỗi `venv\Scripts\activate` không hoạt động, thử:
```bash
python -m venv venv
cd venv\Scripts
activate.bat
cd ../..
```

---

### 3️⃣ **Chuẩn Bị Dữ Liệu & Train Model**

```bash
# CD vào folder DuBao
cd DuBao

# Bước 1: Tiền xử lý dữ liệu (ngày → tháng)
python src/preprocess.py
# ✅ Output: data/monthly_rainfall.csv

# Bước 2: Train mô hình & tạo evaluation
python src/evaluate.py
# ✅ Output: 
#   - models/rainfall_model.pkl
#   - models/predictions_plot.png
#   - models/residuals_plot.png

# Bước 3: Visualize dữ liệu
python src/visualize.py
# ✅ Output:
#   - models/daily_distribution.png
#   - models/monthly_timeseries.png
#   - models/monthly_heatmap.png

# Bước 4: Train mô hình nâng cao (OPTIONAL)
# Nếu muốn so sánh LSTM & ARIMA
python src/advanced_models.py
# ✅ Output:
#   - models/lstm_model.h5
#   - models/arima_model.pkl
#   - Comparison report

# Hoặc chạy tất cả cùng lúc từ main.py:
python main.py
```

**Ghi chú**: 
- Lần đầu chạy `preprocess.py` có thể mất 1-2 phút
- `evaluate.py` tạo biểu đồ, có thể cần Matplotlib backend
- `advanced_models.py` chỉ cần nếu muốn dùng LSTM/ARIMA

---

### 4️⃣ **Setup Django Database**

```bash
# Quay lại thư mục gốc
cd ..

# Tạo database
python manage.py migrate

# Tạo tài khoản admin (tùy chọn, để xem lịch sử)
python manage.py createsuperuser
# Nhập username, email, password

# Check xem OK không
python manage.py check
```

---

### 5️⃣ **Chạy Web Application**

```bash
# Từ thư mục gốc
python manage.py runserver

# Đợi thấy:
# Starting development server at http://127.0.0.1:8000/
# Quit the server with CTRL-BREAK.
```

**Mở browser:**
- http://localhost:8000/ - Dashboard chính
- http://localhost:8000/admin/ - Admin panel (nếu tạo superuser)

---

### 6️⃣ **Sử Dụng Web App**

#### 🏠 Trang chủ (Dashboard)
- Hiển thị thống kê dữ liệu
- Biểu đồ yearly & monthly rainfall
- Form dự đoán

#### 📊 Form Dự Đoán
1. Nhập **Năm** (1979-2100)
2. Nhập **Tháng** (1-12)
3. Nhấn **Dự Đoán**
4. Xem kết quả + metrics

#### 💾 Lịch Sử Dự Đoán
- Đăng nhập để lưu lịch sử
- Xem lại các dự đoán cũ
- Xem thời gian dự đoán

---

## 🎯 Các Biến Thể Chạy

### **Chỉ train model (không web)**
```bash
cd DuBao
python main.py
```

### **Chỉ chạy web (model đã có)**
```bash
python manage.py runserver
```

### **Chạy test evaluation**
```bash
cd DuBao
python src/evaluate.py
# Xem metrics & biểu đồ
```

### **Phân tích dữ liệu**
```bash
cd DuBao
python src/visualize.py
# Xem EDA charts
```

### **So sánh các mô hình**
```bash
cd DuBao
python src/advanced_models.py
# So sánh RF vs LSTM vs ARIMA
```

---

## ⚙️ Troubleshooting

### ❌ **Error: `No module named 'Django'`**
```bash
pip install -r requirements.txt
# hoặc
pip install django pandas numpy scikit-learn matplotlib seaborn statsmodels
```

### ❌ **Error: `sqlite3 database is locked`**
```bash
# Xóa database cũ
del db.sqlite3  # Windows
rm db.sqlite3   # Linux/Mac

# Tạo lại
python manage.py migrate
```

### ❌ **Error: `Connection refused` port 8000**
Django đã chạy ở cửa sổ khác. Nhấn Ctrl+C để dừng, chạy lại.

### ❌ **Error: `data/monthly_rainfall.csv not found`**
```bash
cd DuBao
python src/preprocess.py
cd ..
```

### ❌ **Error: Port 8000 đã được dùng**
```bash
# Chạy ở port khác
python manage.py runserver 8001
# Mở http://localhost:8001
```

### ⚠️ **TensorFlow error (cho LSTM)**
- TensorFlow là tùy chọn, không bắt buộc
- Dự báo vẫn chạy với Random Forest
- Nếu muốn LSTM:
```bash
pip install tensorflow
```

### ⚠️ **Matplotlib backend error**
```bash
# Thêm vào đầu file visualize.py:
import matplotlib
matplotlib.use('Agg')
```

---

## 📁 File Cần Có

**Sau khi setup xong, folder DuBao sẽ có:**
```
DuBao/
├── data/
│   ├── raw_daily.csv              ✅ (có sẵn)
│   └── monthly_rainfall.csv       ✅ (tạo sau preprocess)
├── models/
│   ├── rainfall_model.pkl         ✅ (tạo sau evaluate)
│   ├── predictions_plot.png       ✅
│   ├── residuals_plot.png         ✅
│   ├── daily_distribution.png     ✅ (tạo sau visualize)
│   ├── monthly_timeseries.png     ✅
│   ├── monthly_heatmap.png        ✅
│   ├── lstm_model.h5              ⚠️ (tùy chọn)
│   └── arima_model.pkl            ⚠️ (tùy chọn)
```

---

## 🏃 Chạy Nhanh

**Copy & paste toàn bộ để chạy từ đầu (sau activate venv):**

```bash
# Windows
cd DuBao && python src/preprocess.py && python src/evaluate.py && python src/visualize.py && cd .. && python manage.py migrate && python manage.py runserver

# Linux/Mac
cd DuBao && python3 src/preprocess.py && python3 src/evaluate.py && python3 src/visualize.py && cd .. && python3 manage.py migrate && python3 manage.py runserver
```

Sau đó mở: http://localhost:8000/

---

## 📚 Tài Liệu Chi Tiết

- **README.md** - Thông tin chung về dự án
- **DuBao/main.py** - Chạy pipeline đầy đủ
- **DuBao/src/evaluate.py** - Đánh giá model
- **DuBao/src/visualize.py** - EDA & charts
- **predictor/views.py** - Django views & APIs

---

## ✅ Checklist Cài Đặt

- [ ] Cài Python 3.7+
- [ ] Tạo venv
- [ ] Cài requirements.txt
- [ ] Chạy preprocess.py
- [ ] Chạy evaluate.py
- [ ] Chạy visualize.py
- [ ] Chạy migrate Django
- [ ] Chạy runserver
- [ ] Test web app (localhost:8000)
- [ ] Đăng nhập & test dự đoán

---

## 🎓 Đây Là Gì?

**Đồ án tốt nghiệp** - Dự báo lượng mưa hàng tháng tại Đà Nẵng
- 📊 ML Pipeline: Data preprocessing → Model training → Evaluation
- 🌐 Web App: Django + APIs
- 📈 EDA: Phân tích & visualization
- 🤖 3 Models: Random Forest, LSTM, ARIMA

**Kết quả**: Có thể dự đoán lượng mưa tháng trong tương lai với độ chính xác R² > 0.7

---

**Hãy bắt đầu! 🚀**
