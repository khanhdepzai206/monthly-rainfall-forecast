# Chạy dự án Dự báo Lượng Mưa

## Chuẩn bị (một lần)

```bash
# Từ thư mục gốc dự án
cd "d:\Du Bao Luong Mua"

# Tạo môi trường ảo (khuyến nghị)
python -m venv .venv
.venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt

# Tạo database Django
python manage.py migrate
python manage.py createsuperuser
```

## Cách 1: Chạy web (dùng dữ liệu và mô hình sẵn có)

```bash
cd "d:\Du Bao Luong Mua"
python manage.py runserver
```

Mở trình duyệt: **http://127.0.0.1:8000/**

---

## Cách 2: Cập nhật dữ liệu từ API và train lại mô hình (tăng độ chính xác)

Bước này sẽ:
- Gọi **Open-Meteo API** lấy thêm **cloud_cover**, **surface_pressure** (và temperature, humidity, wind_speed)
- Gộp với lượng mưa → tạo `monthly_combined.csv`
- Train lại 3 mô hình (Gradient Boosting, Random Forest, SARIMA) và lưu R², MAE, RMSE

```bash
cd "d:\Du Bao Luong Mua\DuBao"
python run_pipeline.py --fetch
```

**Lưu ý:** Cần có file `DuBao/data/raw_daily.csv` (dữ liệu lượng mưa ngày). Nếu chưa có `weather_daily.csv`, dùng `--fetch` để tải. API miễn phí, có thể mất vài phút (1979–2022).

Chỉ chuẩn hóa và train lại (không gọi API):

```bash
cd "d:\Du Bao Luong Mua\DuBao"
python run_pipeline.py
```

Sau khi chạy xong pipeline, chạy web như Cách 1 để xem độ chính xác mới trên giao diện.

---

## Chạy từng bước thủ công

```bash
cd "d:\Du Bao Luong Mua\DuBao\src"

# 1) Lấy dữ liệu thời tiết từ API (nhiệt độ, độ ẩm, gió, mây, áp suất)
python fetch_weather_data.py

# 2) Chuyển dữ liệu ngày → tháng và merge với thời tiết
python prepare_data.py

# 3) Train và đánh giá các mô hình
python evaluate_all_models.py
```

---

## Dữ liệu và Feature

- **monthly_combined.csv** cần ít nhất: `year`, `month`, `rainfall`, `temperature`, `humidity`, `wind_speed`.
- Nếu chạy `fetch_weather_data.py` (hoặc `run_pipeline.py --fetch`), file sẽ thêm `cloud_cover`, `surface_pressure` → mô hình dùng thêm feature này và thường **tăng độ chính xác**.

---

## Tóm tắt lệnh chạy dự án

| Mục đích | Lệnh |
|----------|------|
| Chạy web | `cd "d:\Du Bao Luong Mua"` rồi `python manage.py runserver` |
| Cập nhật dữ liệu API + train lại | `cd "d:\Du Bao Luong Mua\DuBao"` rồi `python run_pipeline.py --fetch` |
| Chỉ train lại (không gọi API) | `cd "d:\Du Bao Luong Mua\DuBao"` rồi `python run_pipeline.py` |
