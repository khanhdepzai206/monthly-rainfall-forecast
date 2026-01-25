import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model_with_weather(csv_path, model_path, test_size=0.2, random_state=42):
    """
    Đánh giá mô hình với weather features
    """
    # Load dữ liệu
    df = pd.read_csv(csv_path)
    df = df.sort_values(['year', 'month']).reset_index(drop=True)

    # Feature engineering (giống như trong train_model.py)
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

    df = df.dropna()

    # Tách features
    feature_cols = [col for col in df.columns if col not in ['rainfall', 'year', 'month', 'date']]
    feature_cols.extend(['year', 'month'])

    X = df[feature_cols]
    y = df['rainfall']

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    # Load model
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data.get('scaler')

    # Scale features
    if scaler:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    # Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Calculate metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    print("\n📊 Model Evaluation Results:")
    print("=" * 50)
    print(f"Training Set (n={len(y_train)}):")
    print(f"  MAE: {mae_train:.2f}")
    print(f"  RMSE: {rmse_train:.2f}")
    print(f"  R²: {r2_train:.3f}")
    print(f"\nTest Set (n={len(y_test)}):")
    print(f"  MAE: {mae_test:.2f}")
    print(f"  RMSE: {rmse_test:.2f}")
    print(f"  R²: {r2_test:.3f}")

    return {
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'r2_train': r2_train,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

if __name__ == "__main__":
    # Evaluate mô hình với weather data
    results = evaluate_model_with_weather(
        csv_path="../data/monthly_combined.csv",
        model_path="../models/model_with_weather.pkl"
    )