# 🌐 Trang Web Hiển Thị Độ Chính Xác Mô Hình

Django Server đang chạy. Bạn có thể truy cập các trang sau:

## 📊 **Trang Metrics (Độ Chính Xác)**

```
http://localhost:8000/predictor/model-metrics/
```

**Hiển thị:**
- ✅ Bảng so sánh Classifier Models (Accuracy, Precision, Recall, F1-Score)
- ✅ Bảng so sánh Regressor Models (MAE, RMSE, R², MAPE)
- ✅ Mô hình tốt nhất cho mỗi loại
- ✅ Giải thích từng chỉ số
- ✅ Thông tin test set

---

## 🔮 **Trang Dự Đoán Theo Ngày**

```
http://localhost:8000/predictor/predict-daily/
```

**Chức năng:**
- Nhập năm, tháng, ngày
- Dự đoán: Có mưa hay không?
- Hiển thị: Xác suất mưa + Lượng mưa (mm)
- Dự đoán khoảng 10 ngày

---

## 🔌 **API Endpoints**

### 1. Lấy metrics của tất cả mô hình
```
GET http://localhost:8000/api/model-metrics/
```

**Response:**
```json
{
  "success": true,
  "data": {
    "classifier": {
      "GradientBoosting": {
        "accuracy": 0.84,
        "precision": 0.77,
        "recall": 0.84,
        "f1": 0.81
      }
    },
    "regressor": {
      "GradientBoosting": {
        "mae": 8.87,
        "rmse": 26.54,
        "r2": 0.6421,
        "mape": 14.23
      }
    },
    "test_set_info": {
      "test_count": 2556,
      "rain_count": 1009
    }
  }
}
```

### 2. Dự đoán một ngày
```
POST http://localhost:8000/api/predict/

Body:
{
  "year": 2023,
  "month": 5,
  "day": 15
}
```

### 3. Dự đoán khoảng ngày
```
GET http://localhost:8000/api/predict-range/?year=2023&month=5&start_day=1&num_days=10
```

---

## 📈 **Kết Quả Hiện Tại**

### Classifier (Dự đoán có mưa?):
| Chỉ số | Kết quả |
|--------|--------|
| Accuracy | **84%** |
| F1-Score | **0.81** |
| Precision | **77%** |
| Recall | **84%** |

### Regressor (Dự đoán lượng mưa?):
| Chỉ số | Kết quả |
|--------|--------|
| MAE | **8.87 mm** |
| RMSE | **26.54 mm** |
| R² | **0.6421** |
| MAPE | **14.23%** |

---

## ✅ Các trang khác

```
http://localhost:8000/predictor/                    # Trang chính
http://localhost:8000/predictor/predict/            # Dự đoán theo tháng
http://localhost:8000/predictor/history/            # Lịch sử dự đoán
http://localhost:8000/predictor/comparison/         # So sánh mô hình cũ
```

---

## 🎯 Tiếp theo?

1. **Truy cập trang metrics:**
   ```
   http://localhost:8000/predictor/model-metrics/
   ```

2. **Thử dự đoán:**
   ```
   http://localhost:8000/predictor/predict-daily/
   ```

3. **Test API (trong Postman hoặc cURL):**
   ```bash
   curl http://localhost:8000/api/model-metrics/
   ```

Server đang chạy trên `http://localhost:8000`
