import pandas as pd
import numpy as np
import pickle
import os

def prepare_daily_features(df):
    """Chuẩn bị features cho dữ liệu ngày"""
    df = df.sort_values('date').reset_index(drop=True)
    
    # Cyclical encoding cho month/day
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # Lag features cho rainfall
    df['rainfall_lag_1'] = df['rainfall'].shift(1)
    df['rainfall_lag_3'] = df['rainfall'].shift(3)
    df['rainfall_lag_7'] = df['rainfall'].shift(7)
    df['rainfall_ma_3'] = df['rainfall'].rolling(window=3, min_periods=1).mean()
    df['rainfall_ma_7'] = df['rainfall'].rolling(window=7, min_periods=1).mean()
    df['rainfall_std_7'] = df['rainfall'].rolling(window=7, min_periods=1).std()
    
    # Weather features
    df['temperature_lag_1'] = df['temperature'].shift(1)
    df['temperature_lag_3'] = df['temperature'].shift(3)
    df['temperature_ma_3'] = df['temperature'].rolling(window=3, min_periods=1).mean()
    
    df['humidity_lag_1'] = df['humidity'].shift(1)
    df['humidity_lag_3'] = df['humidity'].shift(3)
    df['humidity_ma_3'] = df['humidity'].rolling(window=3, min_periods=1).mean()
    
    df['wind_speed_lag_1'] = df['wind_speed'].shift(1)
    df['wind_speed_lag_3'] = df['wind_speed'].shift(3)
    df['wind_speed_ma_3'] = df['wind_speed'].rolling(window=3, min_periods=1).mean()
    
    if 'cloud_cover' in df.columns:
        df['cloud_cover_lag_1'] = df['cloud_cover'].shift(1)
        df['cloud_cover_ma_3'] = df['cloud_cover'].rolling(window=3, min_periods=1).mean()
    
    if 'surface_pressure' in df.columns:
        df['surface_pressure_lag_1'] = df['surface_pressure'].shift(1)
        df['surface_pressure_ma_3'] = df['surface_pressure'].rolling(window=3, min_periods=1).mean()
    
    # Trend
    df['trend'] = range(len(df))
    
    return df

def predict_with_best_model(csv_path, year, month, day, models_dir="models"):
    """
    Dự đoán sử dụng mô hình tốt nhất (từ comparison_results)
    """
    
    best_classifier_name = 'GradientBoosting'
    best_regressor_name = 'GradientBoosting'
    
    # Load dữ liệu
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Load Classifier - thử tên mặc định trước
    classifier_path = f"{models_dir}/daily_classifier.pkl"
    if not os.path.exists(classifier_path):
        # Nếu không có, thử tìm comparison results
        try:
            with open(f"{models_dir}/comparison_results.pkl", 'rb') as f:
                comparison = pickle.load(f)
            best_classifier_name = comparison['best_classifier']
            classifier_path = f"{models_dir}/daily_classifier_{best_classifier_name.lower()}.pkl"
        except:
            raise FileNotFoundError(f"Không tìm thấy mô hình. Chạy: python src/train_two_step_daily.py")
    
    with open(classifier_path, 'rb') as f:
        classifier_pkg = pickle.load(f)
    classifier = classifier_pkg['model']
    scaler_c = classifier_pkg['scaler']
    feature_cols = classifier_pkg['features']
    
    # Load Regressor - thử tên mặc định trước
    regressor_path = f"{models_dir}/daily_regressor.pkl"
    if not os.path.exists(regressor_path):
        try:
            with open(f"{models_dir}/comparison_results.pkl", 'rb') as f:
                comparison = pickle.load(f)
            best_regressor_name = comparison['best_regressor']
            regressor_path = f"{models_dir}/daily_regressor_{best_regressor_name.lower()}.pkl"
        except:
            raise FileNotFoundError(f"Không tìm thấy mô hình. Chạy: python src/train_two_step_daily.py")
    
    with open(regressor_path, 'rb') as f:
        regressor_pkg = pickle.load(f)
    regressor = regressor_pkg['model']
    scaler_r = regressor_pkg['scaler']
    
    # Chuẩn bị features
    df_prepared = prepare_daily_features(df.copy())
    df_prepared = df_prepared.dropna()
    
    # Tạo feature vector
    last_row = df_prepared.iloc[-1].copy()
    
    # Cập nhật year, month, day
    last_row['year'] = year
    last_row['month'] = month
    last_row['day'] = day
    last_row['month_sin'] = np.sin(2 * np.pi * month / 12)
    last_row['month_cos'] = np.cos(2 * np.pi * month / 12)
    last_row['day_sin'] = np.sin(2 * np.pi * day / 31)
    last_row['day_cos'] = np.cos(2 * np.pi * day / 31)
    last_row['trend'] = len(df_prepared) + 1
    
    # Lấy features
    X_pred = last_row[feature_cols].values.reshape(1, -1)
    X_pred_scaled = scaler_c.transform(X_pred)
    
    # Dự đoán có mưa hay không
    has_rain_pred = classifier.predict(X_pred_scaled)[0]
    rain_prob = classifier.predict_proba(X_pred_scaled)[0][1]
    
    # Dự đoán lượng mưa (nếu có mưa)
    if has_rain_pred == 1 and rain_prob > 0.3:
        X_pred_r_scaled = scaler_r.transform(X_pred)
        rainfall_pred = regressor.predict(X_pred_r_scaled)[0]
        rainfall_pred = max(rainfall_pred, 0)
    else:
        rainfall_pred = 0
    
    return {
        'year': year,
        'month': month,
        'day': day,
        'has_rain': bool(has_rain_pred),
        'rain_probability': float(rain_prob),
        'predicted_rainfall': float(rainfall_pred),
        'classifier_model': best_classifier_name,
        'regressor_model': best_regressor_name
    }

if __name__ == "__main__":
    year = int(input("Nhập năm: "))
    month = int(input("Nhập tháng (1-12): "))
    day = int(input("Nhập ngày (1-31): "))
    
    result = predict_with_best_model(
        csv_path="data/daily_combined.csv",
        year=year,
        month=month,
        day=day
    )
    
    print(f"\n📅 Dự đoán ngày {day}/{month}/{year}:")
    print(f"  🎯 Classifier: {result['classifier_model']}")
    print(f"  🎯 Regressor: {result['regressor_model']}")
    print(f"  🌦️ Có mưa: {'Có' if result['has_rain'] else 'Không'}")
    print(f"  📊 Xác suất mưa: {result['rain_probability'] * 100:.1f}%")
    print(f"  🌧️ Lượng mưa: {result['predicted_rainfall']:.2f} mm")
