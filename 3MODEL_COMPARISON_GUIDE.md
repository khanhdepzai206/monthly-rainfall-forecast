# 📊 3-Model Comparison Quick Start Guide

## 🎯 What This Feature Does

Compare rainfall predictions from 3 different machine learning models and see their consensus:
- **Gradient Boosting** 🔵
- **Random Forest** 🟢
- **XGBoost** 🟠

Each model independently predicts whether it will rain and how much rainfall to expect.

---

## 🚀 Quick Start

### Option 1: Web Interface (Easiest)

1. **Open the predict page:**
   ```
   http://127.0.0.1:8000/predict/
   ```

2. **Scroll to the "⚔️ So Sánh 3 Mô Hình" section**

3. **Fill in the date:**
   - Year: 2024
   - Month: 5
   - Day: 15

4. **Click "So Sánh" button**

5. **View results:**
   - Table showing all 3 models' predictions
   - Consensus box showing what majority of models predict
   - Color-coded models (blue, green, orange)

### Option 2: API Call

```bash
# Using curl
curl -X POST http://127.0.0.1:8000/api/predict-compare/ \
  -H "Content-Type: application/json" \
  -d '{"year": 2024, "month": 5, "day": 15}'
```

### Option 3: Python Script

```python
import requests

url = "http://127.0.0.1:8000/api/predict-compare/"
data = {"year": 2024, "month": 5, "day": 15}

response = requests.post(url, json=data)
result = response.json()

print(f"Will it rain on {result['date']}?")
print(f"Consensus: {result['consensus']['avg_rain_probability']*100:.1f}% chance")
print(f"Expected rainfall: {result['consensus']['avg_rainfall']:.2f} mm")
```

---

## 📋 Understanding the Results

### Model Prediction Table

| Model | Rain? | Probability | Rainfall | MAE | RMSE | R² |
|-------|-------|-------------|----------|-----|------|-----|
| 🔵 Gradient Boosting | ☔ Có | 70.31% | 0.50 mm | 0.0 | 0.0 | 0.0 |
| 🟢 Random Forest | ☔ Có | 51.50% | 0.19 mm | 0.0 | 0.0 | 0.0 |
| 🟠 XGBoost | ☀️ Không | 0.00% | 0.00 mm | 0.0 | 0.0 | 0.0 |

**How to read:**
- **Rain?** - Will it rain? (☔ Có = Yes, ☀️ Không = No)
- **Probability** - Confidence level (0-100%)
- **Rainfall** - How much rain (in mm)

### Consensus Box

```
📌 Nhận định chung (Consensus)
┌─────────────────────────────────┐
│ ☔ Có mưa (Will Rain)            │
│ Xác suất trung bình: 40.60%      │
│ (Average Probability)            │
│                                 │
│ Lượng mưa dự đoán: 0.23 mm      │
│ (Predicted Rainfall)            │
│                                 │
│ 2/3 mô hình đồng ý             │
│ (Agreement: 2/3 models agree)   │
└─────────────────────────────────┘
```

---

## 🔍 Interpreting Results

### Scenario 1: All Models Agree ✓
```
Agreement: 3/3 models agree
→ High confidence prediction
→ Safe to plan outdoor activities or prepare for rain
```

### Scenario 2: Split Decision 
```
Agreement: 2/3 models agree
→ Moderate confidence
→ Best to check backup plan
```

### Scenario 3: Majority Vote
```
Agreement: 1/3 models agree
→ Low confidence in prediction
→ Models disagree significantly
```

---

## 🎓 What Each Model Does

### 🔵 Gradient Boosting
- Builds trees sequentially, each correcting previous errors
- Good at finding complex patterns
- Often produces middle-ground predictions

### 🟢 Random Forest
- Uses multiple independent decision trees
- Very robust and less prone to overfitting
- Good baseline predictions

### 🟠 XGBoost (eXtreme Gradient Boosting)
- Enhanced gradient boosting algorithm
- Fast and highly optimized
- Often most accurate on test data

---

## 📊 Example Results

### Date: 2024-05-15
```
Model          Has Rain  Probability  Rainfall
─────────────────────────────────────────────
Gradient       YES       70.31%       0.50 mm
Random Forest  YES       51.50%       0.19 mm
XGBoost        NO        0.00%        0.00 mm
─────────────────────────────────────────────
CONSENSUS:     YES       40.60%       0.23 mm
Agreement:     2/3 models say YES
```

**Interpretation:**
- 2 out of 3 models predict rain
- Average probability: 40.60%
- Expected amount: ~0.23 mm (light drizzle)
- **Recommendation:** Bring umbrella ☂️

---

## 🛠️ Technical Details

### Request Format
```json
{
  "year": 2024,
  "month": 5,
  "day": 15
}
```

### Response Format
```json
{
  "success": true,
  "date": "2024-05-15",
  "models": [
    {
      "model": "GradientBoosting",
      "has_rain": true,
      "rain_probability": 0.7031,
      "predicted_rainfall": 0.5,
      "mae": 0.0,
      "rmse": 0.0,
      "r2_score": 0.0
    },
    ...
  ],
  "consensus": {
    "has_rain": true,
    "avg_rain_probability": 0.406,
    "avg_rainfall": 0.23,
    "agreement_count": 2
  }
}
```

---

## ⚠️ Common Issues & Fixes

### Issue: "Lỗi: Không thể so sánh"
**Solution:** 
- Check if server is running
- Verify date is valid (year 1979-2100)
- Check browser console (F12) for error messages

### Issue: Form doesn't respond
**Solution:**
- Refresh page (Ctrl+R)
- Clear browser cache
- Try different date

### Issue: All values are zero
**Solution:**
- This sometimes happens with certain dates
- Try another date
- Check if model files are present

---

## 📈 Using Results for Decision Making

### Weather Planning
```
If consensus prob > 70%
  → Likely to rain, bring umbrella ☂️
If consensus prob 30-70%
  → Uncertain, have backup plan
If consensus prob < 30%
  → Likely to stay dry ☀️
```

### Comparing Models
```
If all 3 models agree
  → Trust the prediction more
If models disagree
  → Be cautious, may need more data
Random Forest (🟢) = most stable predictions
```

---

## 💡 Pro Tips

1. **For Important Events:**
   - Check predictions 3-7 days in advance
   - Watch how consensus changes over time
   - If still uncertain, check meteorological data

2. **Trust the Consensus:**
   - Average of 3 models is more reliable than single model
   - Agreement count shows confidence level

3. **Historical Patterns:**
   - Some dates/months are naturally rainy
   - Check patterns for the specific month/day

---

## 🔗 Related Resources

- Full Prediction: `/predict/` - Make single-day predictions
- Model Metrics: `/metrics/` - View model performance statistics
- API Documentation: See `README.md`
- Training Guide: See `MODEL_COMPARISON_GUIDE.md`

---

**Status:** ✅ Live and Working  
**Last Updated:** 2024  
**Support:** Check server logs or console for errors
