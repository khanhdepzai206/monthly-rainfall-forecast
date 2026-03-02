import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from train_two_step_daily import prepare_daily_features

def predict_daily_rainfall(classifier_path, regressor_path, csv_path, year, month, day):
    """
    Dự đoán lượng mưa theo ngày (2-step):
    Bước 1: Classifier dự đoán có mưa hay không
    Bước 2: Nếu có mưa, Regressor dự đoán lượng mưa
    """
    
    # Load data lịch sử
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Load Classifier
    with open(classifier_path, 'rb') as f:
        classifier_pkg = pickle.load(f)
    classifier = classifier_pkg['model']
    scaler_c = classifier_pkg['scaler']
    feature_cols = classifier_pkg['features']
    
    # Load Regressor
    with open(regressor_path, 'rb') as f:
        regressor_pkg = pickle.load(f)
    regressor = regressor_pkg['model']
    scaler_r = regressor_pkg['scaler']
    
    # Chuẩn bị dữ liệu đến ngày dự đoán
    df_prepared = prepare_daily_features(df.copy())
    
    # Lấy hàng gần nhất có dữ liệu đầy đủ
    df_prepared = df_prepared.dropna()
    
    # Tạo feature vector cho ngày cần dự đoán
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
    
    # ===== BƯỚC 1: DỰ ĐOÁN CÓ MƯA HAY KHÔNG =====
    has_rain_pred = classifier.predict(X_pred_scaled)[0]
    rain_prob = classifier.predict_proba(X_pred_scaled)[0][1]
    
    # ===== BƯỚC 2: DỰ ĐOÁN LƯỢNG MƯA (NẾU CÓ MƯA) =====
    if has_rain_pred == 1 and rain_prob > 0.3:  # Ngưỡng tin cậy
        X_pred_r_scaled = scaler_r.transform(X_pred)
        rainfall_pred = regressor.predict(X_pred_r_scaled)[0]
        rainfall_pred = max(rainfall_pred, 0)  # Không âm
    else:
        rainfall_pred = 0
    
    return {
        'year': year,
        'month': month,
        'day': day,
        'has_rain': bool(has_rain_pred),
        'rain_probability': float(rain_prob),
        'predicted_rainfall': float(rainfall_pred)
    }

def predict_daily_range(classifier_path, regressor_path, csv_path, year, month, start_day=1, num_days=10):
    """Dự đoán mưa cho một khoảng ngày"""
    results = []
    for day in range(start_day, start_day + num_days):
        try:
            result = predict_daily_rainfall(classifier_path, regressor_path, csv_path, year, month, day)
            results.append(result)
        except:
            continue
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test dự đoán
    year = int(input("Nhập năm (ví dụ 2023): "))
    month = int(input("Nhập tháng (1-12): "))
    day = int(input("Nhập ngày (1-31): "))
    
    result = predict_daily_rainfall(
        classifier_path="models/daily_classifier.pkl",
        regressor_path="models/daily_regressor.pkl",
        csv_path="data/daily_combined.csv",
        year=year,
        month=month,
        day=day
    )
    
    print(f"\n📅 Dự đoán ngày {day}/{month}/{year}:")
    print(f"  🌦️ Có mưa: {'Có' if result['has_rain'] else 'Không'}")
    print(f"  📊 Xác suất mưa: {result['rain_probability'] * 100:.1f}%")
    print(f"  🌧️ Lượng mưa dự đoán: {result['predicted_rainfall']:.2f} mm")
