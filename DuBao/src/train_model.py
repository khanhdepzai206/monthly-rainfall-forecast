import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

def train_model(csv_path, model_path):
    """
    Train mô hình cải thiện với Gradient Boosting
    (thay vì Random Forest có sai số cao)
    """
    # Load dữ liệu
    df = pd.read_csv(csv_path)
    df = df.sort_values(['year', 'month']).reset_index(drop=True)
    
    # Feature engineering
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['rainfall_lag_1'] = df['rainfall'].shift(1)
    df['rainfall_lag_12'] = df['rainfall'].shift(12)
    df['rainfall_ma_3'] = df['rainfall'].rolling(window=3, min_periods=1).mean()
    df['rainfall_ma_12'] = df['rainfall'].rolling(window=12, min_periods=1).mean()
    df['trend'] = range(len(df))
    df['quarter'] = (df['month'] - 1) // 3 + 1
    
    # Weather features
    df['temp_lag_1'] = df['temperature'].shift(1)
    df['temp_lag_12'] = df['temperature'].shift(12)
    df['temp_ma_3'] = df['temperature'].rolling(window=3, min_periods=1).mean()
    df['humidity_lag_1'] = df['humidity'].shift(1)
    df['humidity_lag_12'] = df['humidity'].shift(12)
    df['humidity_ma_3'] = df['humidity'].rolling(window=3, min_periods=1).mean()
    df['wind_lag_1'] = df['wind_speed'].shift(1)
    df['wind_lag_12'] = df['wind_speed'].shift(12)
    df['wind_ma_3'] = df['wind_speed'].rolling(window=3, min_periods=1).mean()
    if 'cloud_cover' in df.columns:
        df['cloud_cover_lag_1'] = df['cloud_cover'].shift(1)
        df['cloud_cover_ma_3'] = df['cloud_cover'].rolling(window=3, min_periods=1).mean()
    if 'surface_pressure' in df.columns:
        df['surface_pressure_lag_1'] = df['surface_pressure'].shift(1)
        df['surface_pressure_ma_3'] = df['surface_pressure'].rolling(window=3, min_periods=1).mean()

    df = df.dropna()
    
    # Tách features
    feature_cols = [col for col in df.columns if col not in ['rainfall', 'year', 'month', 'date']]
    feature_cols.extend(['year', 'month'])
    
    X = df[feature_cols]
    y = df['rainfall']
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    # Tạo model Gradient Boosting (tốt hơn Random Forest)
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        subsample=0.8
    )

    # Train
    model.fit(X_scaled, y)

    base_year = int(df['year'].min())
    with open(model_path, "wb") as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'base_year': base_year,
        }, f)

    print(f"Model saved: {model_path}")
    print("✓ Đã train mô hình Gradient Boosting (sai số thấp hơn Random Forest)!")
