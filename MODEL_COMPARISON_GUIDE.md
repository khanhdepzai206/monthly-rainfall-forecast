# 3-Model Comparison Feature - Documentation

## Overview

The system now supports comparing rainfall predictions from 3 different machine learning models simultaneously:
- **GradientBoosting Classifier & Regressor**
- **RandomForest Classifier & Regressor**  
- **XGBoost Classifier & Regressor**

This feature allows you to:
1. Get predictions from all 3 models at once
2. Compare their accuracy metrics (MAE, RMSE, R² Score)
3. See consensus predictions across models
4. Understand model agreement/disagreement

---

## Architecture

### Backend Components

#### 1. **API Endpoint**: `/api/predict-compare/`

**Method**: POST  
**Content-Type**: application/json

**Request Format**:
```json
{
  "year": 2024,
  "month": 5,
  "day": 15
}
```

**Response Format**:
```json
{
  "success": true,
  "date": "2024-05-15",
  "models": [
    {
      "model": "GradientBoosting",
      "has_rain": true,
      "rain_probability": 0.7532,
      "predicted_rainfall": 45.23,
      "mae": 12.4567,
      "rmse": 18.9234,
      "r2_score": 0.8234
    },
    {
      "model": "RandomForest",
      "has_rain": true,
      "rain_probability": 0.6891,
      "predicted_rainfall": 38.15,
      "mae": 13.2145,
      "rmse": 19.8762,
      "r2_score": 0.8012
    },
    {
      "model": "XGBoost",
      "has_rain": false,
      "rain_probability": 0.4523,
      "predicted_rainfall": 22.56,
      "mae": 11.8934,
      "rmse": 17.6543,
      "r2_score": 0.8456
    }
  ],
  "consensus": {
    "has_rain": true,
    "avg_rain_probability": 0.6315,
    "avg_rainfall": 35.31,
    "agreement_count": 2
  }
}
```

**Error Response**:
```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

#### 2. **Core Logic**: `predictor/api_views.py::predict_compare_models_api()`

This function:
- Validates input parameters (year, month, day)
- Loads trained classifier and regressor models for all 3 algorithms
- Prepares feature data from CSV
- Makes predictions using all models
- Calculates accuracy metrics (MAE, RMSE, R² Score)
- Computes consensus metrics
- Returns formatted JSON response

**Key Dependencies**:
- `DuBao/src/predict_all_models.py::predict_with_all_models()` - Core prediction logic
- `DuBao/models/*.pkl` - Trained model files
- `DuBao/data/daily_combined.csv` - Feature data

#### 3. **Model Training**: `DuBao/src/predict_all_models.py`

Contains:
- `prepare_daily_features()` - Feature engineering (lags, moving averages, seasonal encoding)
- `predict_with_all_models()` - Loads models and generates predictions
- Support for classifiers (binary rain/no-rain) and regressors (rainfall amount)

**Required Model Files**:
```
DuBao/models/
├── classifier_gradientboosting.pkl   # Binary classifier
├── classifier_randomforest.pkl
├── classifier_xgboost.pkl
├── regressor_gradientboosting.pkl    # Rainfall amount regressor
├── regressor_randomforest.pkl
└── regressor_xgboost.pkl
```

---

### Frontend Components

#### 1. **HTML Form**: `templates/predict.html` - Lines 155-192

Form section for 3-model comparison with fields:
- Year (number input, 1979-2100)
- Month (dropdown, 1-12)
- Day (number input, 1-31)

#### 2. **JavaScript Handler**: `templates/predict.html` - Lines 350-435

`displayComparisonResult()` function:
- Validates API response
- Parses model predictions and metrics
- Renders comparison table
- Displays consensus information
- Uses color-coding for model distinction

**Model Colors**:
- GradientBoosting: #667eea (blue)
- RandomForest: #48bb78 (green)
- XGBoost: #f6ad55 (orange)

#### 3. **UI Components**:
- Comparison results table (responsive, Bootstrap 5.3.0)
- Consensus alert box showing average metrics
- Agreement count indicator (X/3 models agreed)

---

## Usage Instructions

### For Users (Web Interface)

1. Navigate to `/predict` page
2. Scroll to "⚔️ So Sánh 3 Mô Hình" (Compare 3 Models) section
3. Select Year, Month, Day
4. Click "So Sánh Mô Hình" (Compare Models)
5. View results:
   - **Models Table**: Individual predictions & metrics
   - **Consensus Box**: Average predictions & agreement status

### For Developers (API Usage)

```bash
# Using curl
curl -X POST http://127.0.0.1:8000/api/predict-compare/ \
  -H "Content-Type: application/json" \
  -d '{"year": 2024, "month": 5, "day": 15}'

# Using Python
import requests
response = requests.post(
  'http://127.0.0.1:8000/api/predict-compare/',
  json={"year": 2024, "month": 5, "day": 15}
)
data = response.json()
```

---

## Performance Metrics Explained

### For Each Model:
- **has_rain**: Boolean - Classifier prediction (binary classification)
- **rain_probability**: Float [0-1] - Confidence in rain prediction
- **predicted_rainfall**: Float - Rainfall amount in mm (regressor output)
- **mae**: Mean Absolute Error - Average prediction error
- **rmse**: Root Mean Squared Error - Standard deviation of errors
- **r2_score**: R² Score [0-1] - Model fit quality (higher = better)

### Consensus Metrics:
- **has_rain**: Majority vote across 3 models
- **avg_rain_probability**: Mean probability across models
- **avg_rainfall**: Mean rainfall prediction across models
- **agreement_count**: Number of models predicting rain (0-3)

---

## File Structure

```
.
├── predictor/
│   ├── api_views.py           # Contains predict_compare_models_api()
│   ├── urls.py                 # Maps /api/predict-compare/ route
│   └── views.py                # Django view functions (simplified)
├── templates/
│   └── predict.html            # Contains comparison form & UI
├── DuBao/
│   ├── src/
│   │   └── predict_all_models.py   # Core prediction logic
│   ├── models/
│   │   ├── classifier_*.pkl     # 3 binary classifiers
│   │   └── regressor_*.pkl      # 3 rainfall regressors
│   └── data/
│       └── daily_combined.csv   # Feature data
└── test_compare_api.py          # Test script
```

---

## Testing

### Run Test Script:
```bash
python test_import.py      # Verify imports & model files
python test_compare_api.py # Test API endpoint
```

### Manual Testing:
1. Start Django server: `python manage.py runserver`
2. Visit: http://127.0.0.1:8000/predict
3. Fill form and click "So Sánh Mô Hình"
4. Check browser console for API response details

### Verify Model Files:
```bash
# Check if all models exist
ls -la DuBao/models/classifier_*.pkl DuBao/models/regressor_*.pkl

# Check data file
ls -la DuBao/data/daily_combined.csv
```

---

## Error Handling

### Common Errors and Solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `Model file not found` | Missing `.pkl` files | Train models using `DuBao/src/train_*.py` |
| `Data file not found` | Missing `daily_combined.csv` | Run data preparation pipeline |
| `Invalid date` | Out-of-range year/month/day | Check data availability in CSV |
| `Prediction failed` | Model loading error | Check model file compatibility |
| `500 Internal Server Error` | Backend exception | Check Django logs for details |

---

## Enhancement Ideas

1. **Visualization**: Add charts comparing model predictions
2. **Historical Comparison**: Compare 3-model predictions vs actual rainfall
3. **Model Weights**: Let users vote on model confidence weights
4. **Training Status**: Show when each model was last trained
5. **Batch Prediction**: Support predicting multiple days at once
6. **Model Retraining**: Add UI to retrain models with new data

---

## Related Documentation

- [Daily Prediction Guide](./DAILY_PREDICTION_GUIDE.md)
- [Web Metrics Guide](./WEB_METRICS_GUIDE.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Project README](./README.md)

---

**Last Updated**: 2024  
**Status**: Production Ready  
**Feature Version**: 1.0
