import pickle
import pandas as pd
import numpy as np

def predict_rainfall(model_path, year, month, weather_data_path=None):
    """
    Dự đoán lượng mưa sử dụng mô hình đã train
    Hỗ trợ cả Gradient Boosting và SARIMA
    """
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
# Check model type
    if isinstance(data, dict) and 'model' in data:
        model_obj = data['model']
        if hasattr(model_obj, 'get_forecast'):  # SARIMA/SARIMAX
            try:
                # SARIMA chỉ dự đoán giá trị tiếp theo, không phải cho tháng cụ thể
                # Thay vào đó, tính seasonal average cho tháng đó từ dữ liệu lịch sử
                if weather_data_path:
                    weather_df = pd.read_csv(weather_data_path)
                    # Tính trung bình rainfall cho tháng đó từ tất cả các năm
                    monthly_avg = weather_df[weather_df['month'] == month]['rainfall'].mean()
                    prediction = monthly_avg if not pd.isna(monthly_avg) else 100
                else:
                    # Fallback nếu không có weather data
                    prediction = 100
            except Exception as e:
                print(f"SARIMA prediction error: {e}")
                prediction = 100  # fallback
        elif hasattr(model_obj, 'predict'):  # sklearn models
            scaler = data.get('scaler', None)
            feature_cols = data.get('feature_cols', ["year", "month"])
            
            # Load weather data for historical averages
            weather_features = {}
            if weather_data_path and 'temperature' in feature_cols:
                weather_df = pd.read_csv(weather_data_path)
                monthly_avg = weather_df.groupby('month').agg({
                    'temperature': 'mean',
                    'humidity': 'mean', 
                    'wind_speed': 'mean'
                }).loc[month]
                
                weather_features = {
                    'temperature': monthly_avg['temperature'],
                    'temp_lag_1': monthly_avg['temperature'],
                    'temp_lag_12': monthly_avg['temperature'],
                    'temp_ma_3': monthly_avg['temperature'],
                    'humidity': monthly_avg['humidity'],
                    'humidity_lag_1': monthly_avg['humidity'],
                    'humidity_lag_12': monthly_avg['humidity'],
                    'humidity_ma_3': monthly_avg['humidity'],
                    'wind_speed': monthly_avg['wind_speed'],
                    'wind_lag_1': monthly_avg['wind_speed'],
                    'wind_lag_12': monthly_avg['wind_speed'],
                    'wind_ma_3': monthly_avg['wind_speed']
                }
            
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
                'quarter': (month - 1) // 3 + 1,
                **weather_features
            }
            
            X = pd.DataFrame({col: [features.get(col, 0)] for col in feature_cols})
            
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X.values
            
            prediction = model_obj.predict(X_scaled)[0]
        else:
            # Old Random Forest model
            X = pd.DataFrame({"year": [year], "month": [month]})
            prediction = model_obj.predict(X)[0]
    else:
        # Direct model object (legacy)
        if hasattr(data, 'get_forecast'):  # SARIMA
            forecast = data.get_forecast(steps=1)
            prediction = forecast.predicted_mean.values[0]
        else:
            prediction = data.predict(pd.DataFrame({"year": [year], "month": [month]}))[0]
    
    return max(0, prediction)  # Không âm
    
    return max(0, prediction)  # Không âm

