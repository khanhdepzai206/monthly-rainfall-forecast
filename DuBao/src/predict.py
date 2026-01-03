import pickle
import pandas as pd
import numpy as np

def predict_rainfall(model_path, year, month):
    """
    Dự đoán lượng mưa sử dụng mô hình đã train
    Hỗ trợ cả Gradient Boosting và SARIMA
    """
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    # Check if it's SARIMA or Gradient Boosting
    if hasattr(data, 'get_forecast'):  # SARIMA
        # SARIMA model
        # Dự đoán 1 bước tiếp theo
        forecast = data.get_forecast(steps=1)
        prediction = forecast.predicted_mean.values[0]
        
    elif isinstance(data, dict):  # Gradient Boosting
        # Gradient Boosting with scaler
        model = data['model']
        scaler = data.get('scaler', None)
        feature_cols = data.get('feature_cols', ["year", "month"])
        
        features = {
            'year': year,
            'month': month,
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'rainfall_lag_1': 0,
            'rainfall_lag_12': 0,
            'rainfall_ma_3': 0,
            'rainfall_ma_12': 0,
            'trend': 0,
            'quarter': (month - 1) // 3 + 1
        }
        
        X = pd.DataFrame({col: [features.get(col, 0)] for col in feature_cols})
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        prediction = model.predict(X_scaled)[0]
    else:
        # Old Random Forest model (không có scaling)
        X = pd.DataFrame({"year": [year], "month": [month]})
        prediction = data.predict(X)[0]
    
    return max(0, prediction)  # Không âm

