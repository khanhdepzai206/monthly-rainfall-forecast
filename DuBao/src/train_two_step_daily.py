import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

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

def train_two_step_daily_model(csv_path, classifier_path, regressor_path):
    """
    Train 2 models:
    1. Classifier: Dự đoán có mưa hay không (rainfall > 0)
    2. Regressor: Dự đoán lượng mưa (chỉ cho các ngày có mưa)
    """
    print("📊 Đang load dữ liệu ngày...")
    df = pd.read_csv(csv_path)
    
    # Chuẩn bị features
    df = prepare_daily_features(df)
    df = df.dropna()
    
    print(f"✓ Loaded {len(df)} ngày dữ liệu")
    
    # Định nghĩa target
    df['has_rain'] = (df['rainfall'] > 0).astype(int)
    
    print(f"  - Ngày có mưa: {(df['has_rain'] == 1).sum()} ({(df['has_rain'] == 1).sum() / len(df) * 100:.1f}%)")
    print(f"  - Ngày không mưa: {(df['has_rain'] == 0).sum()} ({(df['has_rain'] == 0).sum() / len(df) * 100:.1f}%)")
    
    # Chọn features
    feature_cols = [col for col in df.columns if col not in ['rainfall', 'has_rain', 'year', 'month', 'day', 'date']]
    X = df[feature_cols]
    
    # ===== BƯỚC 1: TRAIN CLASSIFIER =====
    print("\n🎯 Huấn luyện Classifier (dự đoán có mưa hay không)...")
    y_classifier = df['has_rain']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_classifier, test_size=0.2, random_state=42, stratify=y_classifier
    )
    
    # Scale dữ liệu
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)
    
    # Train Classifier
    classifier = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    classifier.fit(X_train_c_scaled, y_train_c)
    
    # Đánh giá Classifier
    y_pred_c = classifier.predict(X_test_c_scaled)
    y_pred_proba_c = classifier.predict_proba(X_test_c_scaled)[:, 1]
    
    print("\n📈 Kết quả Classifier:")
    print(classification_report(y_test_c, y_pred_c, target_names=['No Rain', 'Rain']))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_c, y_pred_c)}")
    
    # Lưu Classifier
    with open(classifier_path, 'wb') as f:
        pickle.dump({'model': classifier, 'scaler': scaler_c, 'features': feature_cols}, f)
    print(f"\n✅ Đã lưu Classifier: {classifier_path}")
    
    # ===== BƯỚC 2: TRAIN REGRESSOR (chỉ cho ngày có mưa) =====
    print("\n🌧️ Huấn luyện Regressor (dự đoán lượng mưa)...")
    
    # Chỉ lấy ngày có mưa
    rain_mask = df['has_rain'] == 1
    X_rain = X[rain_mask]
    y_rain = df[rain_mask]['rainfall']
    
    print(f"✓ Dữ liệu huấn luyện Regressor: {len(X_rain)} ngày có mưa")
    
    if len(X_rain) > 100:  # Cần đủ dữ liệu
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_rain, y_rain, test_size=0.2, random_state=42
        )
        
        # Scale dữ liệu
        scaler_r = StandardScaler()
        X_train_r_scaled = scaler_r.fit_transform(X_train_r)
        X_test_r_scaled = scaler_r.transform(X_test_r)
        
        # Train Regressor
        regressor = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        regressor.fit(X_train_r_scaled, y_train_r)
        
        # Đánh giá Regressor
        y_pred_r = regressor.predict(X_test_r_scaled)
        mae = mean_absolute_error(y_test_r, y_pred_r)
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
        r2 = r2_score(y_test_r, y_pred_r)
        
        print("\n📈 Kết quả Regressor (chỉ ngày có mưa):")
        print(f"  MAE: {mae:.2f} mm")
        print(f"  RMSE: {rmse:.2f} mm")
        print(f"  R²: {r2:.4f}")
        
        # Lưu Regressor
        with open(regressor_path, 'wb') as f:
            pickle.dump({'model': regressor, 'scaler': scaler_r, 'features': feature_cols}, f)
        print(f"\n✅ Đã lưu Regressor: {regressor_path}")
    else:
        print("⚠️ Không đủ dữ liệu để huấn luyện Regressor!")

if __name__ == "__main__":
    train_two_step_daily_model(
        csv_path="data/daily_combined.csv",
        classifier_path="models/daily_classifier.pkl",
        regressor_path="models/daily_regressor.pkl"
    )
