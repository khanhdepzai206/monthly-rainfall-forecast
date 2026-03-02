# QUICK REFERENCE - 3 Model Comparison

## 🚀 Start Server
```bash
cd "d:\Du Bao Luong Mua"
python manage.py runserver 0.0.0.0:8000
```

## 🌐 Access Web Interface
```
http://127.0.0.1:8000/predict/
```
Scroll to "⚔️ So Sánh 3 Mô Hình" section

## 📡 Test API
```bash
# Using curl
curl -X POST http://127.0.0.1:8000/api/predict-compare/ \
  -H "Content-Type: application/json" \
  -d '{"year": 2024, "month": 5, "day": 15}'

# Using Python
python -c "
import requests
r = requests.post('http://127.0.0.1:8000/api/predict-compare/', 
                  json={'year': 2024, 'month': 5, 'day': 15})
print(r.json())
"
```

## 🧪 Run Tests
```bash
python test_complete_feature.py
```

## 📊 Result Interpretation

| Model | Rain? | Probability | Rainfall |
|-------|-------|-------------|----------|
| 🔵 GB | ☔/☀️ | 0-100% | mm |
| 🟢 RF | ☔/☀️ | 0-100% | mm |
| 🟠 XGB | ☔/☀️ | 0-100% | mm |

**Consensus:** Average of 3 models + agreement count

## 📁 Key Files
- API: `predictor/api_views.py`
- Logic: `DuBao/src/predict_all_models.py`
- UI: `templates/predict.html`
- Tests: `test_complete_feature.py`

## 📚 Documentation
- `COMPLETION_REPORT.md` - Full report
- `FEATURE_STATUS.md` - Feature details
- `3MODEL_COMPARISON_GUIDE.md` - User guide
- `STATUS_REPORT.txt` - Executive summary

## ⚙️ Model Info
- **GradientBoosting** (🔵): Sequential tree-based
- **RandomForest** (🟢): Ensemble method
- **XGBoost** (🟠): Optimized gradient boosting

## 🔧 Troubleshooting
- Check `server.log` for errors
- View browser console (F12) for JavaScript errors
- Verify all 6 model files in `DuBao/models/`
- Confirm CSV exists at `DuBao/data/daily_combined.csv`

## ✅ Status
- **Tests:** 3/3 PASSING
- **API:** 200 OK, Valid JSON
- **UI:** Fully functional
- **Models:** All 3 working
- **Deployment:** READY

---
**All systems operational. Ready to use! 🎉**
