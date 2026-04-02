#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rolling forecast cho 4-5 ngày tiếp theo dựa trên:
- Dữ liệu lịch sử đến 01/04/2026
- Dữ liệu thời tiết dự báo từ Open-Meteo forecast API
"""
import requests
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

def fetch_forecast_weather(start_date, num_days=5, lat=16.0678, lon=108.2208):
    """
    Fetch dữ liệu thời tiết dự báo từ Open-Meteo.
    Dữ liệu có sẵn cho 7-10 ngày tiếp theo.
    """
    end_date = (pd.to_datetime(start_date) + timedelta(days=num_days)).strftime('%Y-%m-%d')
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
            "cloud_cover_mean",
            "surface_pressure_mean",
        ],
        "timezone": "Asia/Ho_Chi_Minh",
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            daily = data["daily"]
            
            forecast_df = pd.DataFrame({
                "date": pd.to_datetime(daily["time"]),
                "temperature": (np.array(daily["temperature_2m_max"]) + np.array(daily["temperature_2m_min"])) / 2,
                "humidity": daily["relative_humidity_2m_mean"],
                "wind_speed": daily["wind_speed_10m_max"],
                "cloud_cover": daily["cloud_cover_mean"],
                "surface_pressure": daily["surface_pressure_mean"],
            })
            return forecast_df
        else:
            print(f"API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching forecast: {e}")
        return None


def rolling_forecast(start_date, num_days=5, models_dir='DuBao/models', data_dir='DuBao/data'):
    """
    Dự báo rolling cho num_days ngày tiếp theo.
    
    Quy trình:
    1. Đọc dữ liệu lịch sử (historical data)
    2. Fetch forecast thời tiết
    3. Dự báo ngày đầu tiên dựa trên historical lags
    4. Dùng dự đoán đó + forecast thời tiết để dự báo ngày tiếp theo
    5. Lặp lại 5 lần
    """
    
    print(f"📅 Rolling forecast từ {start_date} cho {num_days} ngày")
    
    # Load historical data
    historical_path = os.path.join(data_dir, 'daily_features.csv')
    df_hist = pd.read_csv(historical_path)
    df_hist['date'] = pd.to_datetime(df_hist['date']).dt.date
    
    # Load models
    models_data = {
        'RF': joblib.load(os.path.join(models_dir, 'rf_daily_model.pkl')),
        'XGB': joblib.load(os.path.join(models_dir, 'xgb_daily_model.pkl')),
        'LR': joblib.load(os.path.join(models_dir, 'lr_daily_model.pkl')),
    }
    
    # Fetch forecast weather
    start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    forecast_df = fetch_forecast_weather(start_date_str, num_days)
    
    if forecast_df is None:
        print("❌ Không lấy được dữ liệu forecast thời tiết")
        return None
    
    # Prepare initial lag features from last historical day
    last_hist_date = pd.to_datetime(df_hist['date'].max()).date()
    last_hist_row = df_hist[df_hist['date'] == last_hist_date].iloc[0]
    
    # Lag features cần thiết
    lag_features = [
        'temperature_lag_1', 'temperature_lag_2', 'temperature_lag_3',
        'temperature_lag_4', 'temperature_lag_5', 'temperature_lag_6',
        'temperature_lag_7', 'humidity_lag_1', 'humidity_lag_2', 'humidity_lag_3',
        'humidity_lag_4', 'humidity_lag_5', 'humidity_lag_6', 'humidity_lag_7',
        'wind_speed_lag_1', 'wind_speed_lag_2', 'wind_speed_lag_3',
        'wind_speed_lag_4', 'wind_speed_lag_5', 'wind_speed_lag_6',
        'wind_speed_lag_7', 'cloud_cover_lag_1', 'cloud_cover_lag_2',
        'cloud_cover_lag_3', 'cloud_cover_lag_4', 'cloud_cover_lag_5',
        'cloud_cover_lag_6', 'cloud_cover_lag_7', 'surface_pressure_lag_1',
        'surface_pressure_lag_2', 'surface_pressure_lag_3',
        'surface_pressure_lag_4', 'surface_pressure_lag_5',
        'surface_pressure_lag_6', 'surface_pressure_lag_7', 'rainfall_lag_1',
        'rainfall_lag_2', 'rainfall_lag_3', 'rainfall_lag_4', 'rainfall_lag_5',
        'rainfall_lag_6', 'rainfall_lag_7'
    ]
    
    # Initialize lag history (từ historical data, shift 1 step forward)
    lag_history = {}
    for feature in lag_features:
        lag_history[feature] = last_hist_row[feature]
    
    # Rolling predictions
    predictions = []
    current_temp = last_hist_row['temperature']
    current_humidity = last_hist_row['humidity']
    current_wind = last_hist_row['wind_speed']
    current_cloud = last_hist_row['cloud_cover']
    current_pressure = last_hist_row['surface_pressure']
    current_rainfall = last_hist_row['rainfall']
    
    for i, forecast_row in forecast_df.iterrows():
        pred_date = forecast_row['date'].date()
        forecast_temp = forecast_row['temperature']
        forecast_humidity = forecast_row['humidity']
        forecast_wind = forecast_row['wind_speed']
        forecast_cloud = forecast_row['cloud_cover']
        forecast_pressure = forecast_row['surface_pressure']
        
        # Update lag features với forecast thời tiết
        # Shift lags
        new_lags = lag_history.copy()
        
        new_lags['temperature_lag_1'] = forecast_temp
        new_lags['temperature_lag_2'] = lag_history['temperature_lag_1']
        new_lags['temperature_lag_3'] = lag_history['temperature_lag_2']
        new_lags['temperature_lag_4'] = lag_history['temperature_lag_3']
        new_lags['temperature_lag_5'] = lag_history['temperature_lag_4']
        new_lags['temperature_lag_6'] = lag_history['temperature_lag_5']
        new_lags['temperature_lag_7'] = lag_history['temperature_lag_6']
        
        new_lags['humidity_lag_1'] = forecast_humidity
        new_lags['humidity_lag_2'] = lag_history['humidity_lag_1']
        new_lags['humidity_lag_3'] = lag_history['humidity_lag_2']
        new_lags['humidity_lag_4'] = lag_history['humidity_lag_3']
        new_lags['humidity_lag_5'] = lag_history['humidity_lag_4']
        new_lags['humidity_lag_6'] = lag_history['humidity_lag_5']
        new_lags['humidity_lag_7'] = lag_history['humidity_lag_6']
        
        new_lags['wind_speed_lag_1'] = forecast_wind
        new_lags['wind_speed_lag_2'] = lag_history['wind_speed_lag_1']
        new_lags['wind_speed_lag_3'] = lag_history['wind_speed_lag_2']
        new_lags['wind_speed_lag_4'] = lag_history['wind_speed_lag_3']
        new_lags['wind_speed_lag_5'] = lag_history['wind_speed_lag_4']
        new_lags['wind_speed_lag_6'] = lag_history['wind_speed_lag_5']
        new_lags['wind_speed_lag_7'] = lag_history['wind_speed_lag_6']
        
        new_lags['cloud_cover_lag_1'] = forecast_cloud
        new_lags['cloud_cover_lag_2'] = lag_history['cloud_cover_lag_1']
        new_lags['cloud_cover_lag_3'] = lag_history['cloud_cover_lag_2']
        new_lags['cloud_cover_lag_4'] = lag_history['cloud_cover_lag_3']
        new_lags['cloud_cover_lag_5'] = lag_history['cloud_cover_lag_4']
        new_lags['cloud_cover_lag_6'] = lag_history['cloud_cover_lag_5']
        new_lags['cloud_cover_lag_7'] = lag_history['cloud_cover_lag_6']
        
        new_lags['surface_pressure_lag_1'] = forecast_pressure
        new_lags['surface_pressure_lag_2'] = lag_history['surface_pressure_lag_1']
        new_lags['surface_pressure_lag_3'] = lag_history['surface_pressure_lag_2']
        new_lags['surface_pressure_lag_4'] = lag_history['surface_pressure_lag_3']
        new_lags['surface_pressure_lag_5'] = lag_history['surface_pressure_lag_4']
        new_lags['surface_pressure_lag_6'] = lag_history['surface_pressure_lag_5']
        new_lags['surface_pressure_lag_7'] = lag_history['surface_pressure_lag_6']
        
        # Rainfall lags sẽ dùng dự đoán từ lần trước
        new_lags['rainfall_lag_1'] = current_rainfall
        new_lags['rainfall_lag_2'] = lag_history['rainfall_lag_1']
        new_lags['rainfall_lag_3'] = lag_history['rainfall_lag_2']
        new_lags['rainfall_lag_4'] = lag_history['rainfall_lag_3']
        new_lags['rainfall_lag_5'] = lag_history['rainfall_lag_4']
        new_lags['rainfall_lag_6'] = lag_history['rainfall_lag_5']
        new_lags['rainfall_lag_7'] = lag_history['rainfall_lag_6']
        
        # Tạo feature vector để dự đoán
        feature_vector = pd.DataFrame([new_lags])
        
        # Dự đoán với 3 models
        model_preds = {}
        for model_key, model in models_data.items():
            try:
                pred = float(model.predict(feature_vector)[0])
                model_preds[model_key] = max(0, pred)  # Rainfall không âm
            except Exception as e:
                print(f"  Error predicting with {model_key}: {e}")
                model_preds[model_key] = 0
        
        predictions.append({
            'date': pred_date,
            'date_str': pred_date.strftime('%d/%m/%Y'),
            'RF': round(model_preds['RF'], 2),
            'XGB': round(model_preds['XGB'], 2),
            'LR': round(model_preds['LR'], 2),
            'forecast_temp': round(forecast_temp, 1),
            'forecast_humidity': round(forecast_humidity, 1),
            'forecast_wind': round(forecast_wind, 1),
        })
        
        print(f"  ✓ {pred_date}: RF={model_preds['RF']:.2f}mm, XGB={model_preds['XGB']:.2f}mm, LR={model_preds['LR']:.2f}mm")
        
        # Update lag history cho ngày tiếp theo
        lag_history = new_lags.copy()
        current_rainfall = model_preds['RF']  # Dùng RF làm actual rainfall
        current_temp = forecast_temp
        current_humidity = forecast_humidity
        current_wind = forecast_wind
        current_cloud = forecast_cloud
        current_pressure = forecast_pressure
    
    return predictions


if __name__ == "__main__":
    # Test: dự báo từ 02/04/2026 (ngày sau 01/04)
    predictions = rolling_forecast('2026-04-02', num_days=5)
    
    if predictions:
        print("\n" + "=" * 60)
        print("🎉 ROLLING FORECAST RESULTS")
        print("=" * 60)
        for pred in predictions:
            print(f"{pred['date_str']}: RF={pred['RF']}mm, XGB={pred['XGB']}mm, LR={pred['LR']}mm")
