# Quick Start: 3-Model Rainfall Comparison

## 🎯 5-Minute Setup

### Step 1: Verify Python Environment
```bash
cd "d:\Du Bao Luong Mua"
python --version  # Should be 3.10+
```

### Step 2: Install Requirements (if needed)
```bash
pip install -r requirements.txt
```

### Step 3: Start Django Server
```bash
python manage.py runserver
```
- Server will start at: **http://127.0.0.1:8000**
- Look for message: `Starting development server at http://127.0.0.1:8000/`

### Step 4: Open Prediction Page
- Visit: **http://127.0.0.1:8000/predict**
- Scroll down to: **"⚔️ So Sánh 3 Mô Hình"** section

### Step 5: Try a Prediction
1. Enter **Year**: 2024
2. Select **Month**: 5 (May)
3. Enter **Day**: 15
4. Click **"So Sánh Mô Hình"** button
5. Wait 2-3 seconds for results

---

## 📊 What You'll See

### Results Table
```
| Mô Hình           | Có Mưa | Xác Suất | Lượng Mưa | MAE     | RMSE    | R² Score |
|-------------------|--------|----------|-----------|---------|---------|----------|
| ● GradientBoosting | ☔ Có   | 75.3%    | 45.23 mm  | 0.1245  | 0.1892  | 0.8234   |
| ● RandomForest     | ☔ Có   | 68.9%    | 38.15 mm  | 0.1321  | 0.1988  | 0.8012   |
| ● XGBoost          | ☀️ Không | 45.2%   | 22.56 mm  | 0.1189  | 0.1765  | 0.8456   |
```

### Consensus Box
```
Kết Quả Đồng Thuận
├─ Có Mưa: ☔ Có mưa
├─ Xác Suất Trung Bình: 63.1%
├─ Lượng Mưa TB: 35.31 mm
└─ Sự Thống Nhất: 2/3 mô hình đồng ý
```

---

## 🎓 Understanding the Results

### Rain Predictions
- **☔ Có mưa** = Model predicts RAIN
- **☀️ Không mưa** = Model predicts NO RAIN

### Accuracy Metrics
- **MAE** (Mean Absolute Error): Lower = More Accurate
- **RMSE** (Root Mean Square Error): Lower = More Accurate  
- **R² Score**: Higher (closer to 1.0) = Better fit

### Consensus
- **Agreement Count**: How many models predicted the same result
  - 3/3 = All models agree (high confidence)
  - 2/3 = Majority agree (medium confidence)
  - 1/3 = Disagreement (low confidence)

---

## 📱 Browser Compatibility

✅ Chrome 90+  
✅ Firefox 88+  
✅ Safari 14+  
✅ Edge 90+  

**Recommended**: Chrome or Edge for best experience

---

## ⚙️ System Requirements

- ✅ Python 3.10+
- ✅ Django 4.2+
- ✅ 2GB RAM minimum
- ✅ 500MB disk space
- ✅ Windows/Mac/Linux supported

---

## 🔧 Troubleshooting

### Problem: "Connection refused" at step 3
**Solution**: 
- Make sure port 8000 is not in use
- Try: `python manage.py runserver 8001` (different port)

### Problem: Form submits but no results
**Solution**:
1. Open Browser Console: Press `F12`
2. Go to Console tab
3. Check for error messages
4. Verify date is between 1979-2024

### Problem: "Model not found" error
**Solution**:
- Run: `python test_import.py`
- Check output for missing model files

---

## 📞 API for Developers

### Test Endpoint
```bash
curl -X POST http://127.0.0.1:8000/api/predict-compare/ \
  -H "Content-Type: application/json" \
  -d '{"year": 2024, "month": 5, "day": 15}'
```

### Response Format
```json
{
  "success": true,
  "date": "2024-05-15",
  "models": [...],
  "consensus": {...}
}
```

---

## 📚 Full Documentation

For detailed information:
- [MODEL_COMPARISON_GUIDE.md](MODEL_COMPARISON_GUIDE.md) - Complete feature guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
- [README.md](README.md) - Project overview

---

## 🚀 Common Tasks

### Change Server Port
```bash
python manage.py runserver 8001  # Use port 8001 instead
```

### Test API Manually
```bash
python test_compare_api.py
```

### Verify Models Exist
```bash
python test_import.py
```

### Stop Server
- Press: `Ctrl + C` in terminal

---

## 💡 Tips

1. **Bookmark the page**: http://127.0.0.1:8000/predict
2. **Try different dates**: Models may predict better for certain seasons
3. **Check agreement**: Higher agreement = More confident prediction
4. **Compare accuracy**: Look at MAE/RMSE to see which model performs best
5. **Use ensemble**: Average of 3 models often better than single model

---

## 📞 Support

If you encounter issues:
1. Check [MODEL_COMPARISON_GUIDE.md](MODEL_COMPARISON_GUIDE.md) troubleshooting section
2. Run `python test_import.py` to verify setup
3. Check Django logs for error messages
4. Verify all files exist: `python test_import.py`

---

**Status**: ✅ Ready to Use  
**Version**: 1.0  
**Last Updated**: 2024
