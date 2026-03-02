"""
Predict with multiple models and compare results
"""
import pandas as pd
import numpy as np
import pickle
import os

def prepare_daily_features(df):
    """Chuẩn bị features cho dữ liệu ngày"""
    df = df.sort_values('date').reset_index(drop=True)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    df['rainfall_lag_1'] = df['rainfall'].shift(1)
    df['rainfall_lag_3'] = df['rainfall'].shift(3)
    df['rainfall_lag_7'] = df['rainfall'].shift(7)
    df['rainfall_ma_3'] = df['rainfall'].rolling(window=3, min_periods=1).mean()
    df['rainfall_ma_7'] = df['rainfall'].rolling(window=7, min_periods=1).mean()
    df['rainfall_std_7'] = df['rainfall'].rolling(window=7, min_periods=1).std().fillna(0)
    
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
    
    df['trend'] = range(len(df))
    
    # Fill NaN values
    df = df.fillna(df.mean(numeric_only=True))
    
    return df

def predict_with_all_models(csv_path, year, month, day, models_dir="models"):
    """
    Dự đoán với tất cả mô hình (GradientBoosting, RandomForest, XGBoost)
    Return kết quả từ tất cả mô hình
    """
    
    predictions = {}
    
    for model_name in ['GradientBoosting', 'RandomForest', 'XGBoost']:
        try:
            # Load classifier
            classifier_path = os.path.join(models_dir, f"classifier_{model_name.lower()}.pkl")
            with open(classifier_path, 'rb') as f:
                classifier_pkg = pickle.load(f)
            
            # Load regressor
            regressor_path = os.path.join(models_dir, f"regressor_{model_name.lower()}.pkl")
            with open(regressor_path, 'rb') as f:
                regressor_pkg = pickle.load(f)
            
            # Get feature columns from model
            feature_cols = classifier_pkg.get('features', [])
            if not feature_cols:
                # Fallback: try to infer from model
                feature_cols = [col for col in ['year', 'month', 'day', 'rainfall', 'temperature', 'humidity', 
                                                 'wind_speed', 'rainfall_lag_1', 'rainfall_lag_3', 'rainfall_lag_7',
                                                 'rainfall_ma_3', 'rainfall_ma_7', 'rainfall_std_7',
                                                 'temperature_lag_1', 'temperature_lag_3', 'temperature_ma_3',
                                                 'humidity_lag_1', 'humidity_lag_3', 'humidity_ma_3',
                                                 'wind_speed_lag_1', 'wind_speed_lag_3', 'wind_speed_ma_3',
                                                 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'trend']]
            
            # Load and prepare data
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Prepare features
            df_prepared = prepare_daily_features(df.copy())
            
            # Get last row and update with new date
            last_row = df_prepared.iloc[-1].copy()
            last_row['year'] = year
            last_row['month'] = month
            last_row['day'] = day
            last_row['month_sin'] = np.sin(2 * np.pi * month / 12)
            last_row['month_cos'] = np.cos(2 * np.pi * month / 12)
            last_row['day_sin'] = np.sin(2 * np.pi * day / 31)
            last_row['day_cos'] = np.cos(2 * np.pi * day / 31)
            last_row['trend'] = len(df_prepared) + 1
            
            # Prepare feature vector - only use features the model was trained with
            X_pred = []
            for col in feature_cols:
                if col in last_row.index:
                    X_pred.append(float(last_row[col]))
                else:
                    X_pred.append(0.0)
            X_pred = np.array(X_pred).reshape(1, -1)
            
            # Scale and predict
            classifier = classifier_pkg['model']
            scaler_c = classifier_pkg['scaler']
            regressor = regressor_pkg['model']
            scaler_r = regressor_pkg['scaler']
            
            X_pred_scaled = scaler_c.transform(X_pred)
            has_rain_pred = classifier.predict(X_pred_scaled)[0]
            rain_prob = float(classifier.predict_proba(X_pred_scaled)[0][1])
            
            if has_rain_pred == 1:
                X_pred_r_scaled = scaler_r.transform(X_pred)
                rainfall_pred = float(regressor.predict(X_pred_r_scaled)[0])
                rainfall_pred = max(rainfall_pred, 0)
            else:
                rainfall_pred = 0.0
            
            # Get metrics
            mae = float(regressor_pkg.get('metrics', {}).get('mae', 0))
            rmse = float(regressor_pkg.get('metrics', {}).get('rmse', 0))
            r2_score = float(regressor_pkg.get('metrics', {}).get('r2_score', 0))
            
            predictions[model_name] = {
                'has_rain': bool(has_rain_pred == 1),
                'rain_probability': float(rain_prob),
                'predicted_rainfall': float(rainfall_pred),
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2_score
            }
        except Exception as e:
            # Fallback: return zero values with error message
            import traceback
            print(f"Error in {model_name}: {str(e)}")
            print(traceback.format_exc())
            predictions[model_name] = {
                'has_rain': False,
                'rain_probability': 0.0,
                'predicted_rainfall': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'r2_score': 0.0
            }
    
    return predictions

def predict_with_all_models(csv_path, year, month, day, models_dir="models"):
    """
    Dự đoán với tất cả mô hình (GradientBoosting, RandomForest, XGBoost)
    Return kết quả từ tất cả mô hình
    """
    
    # Load dữ liệu
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Chuẩn bị features
    df_prepared = prepare_daily_features(df.copy())
    df_prepared = df_prepared.dropna()
    
    # Tạo feature vector
    last_row = df_prepared.iloc[-1].copy()
    last_row['year'] = year
    last_row['month'] = month
    last_row['day'] = day
    last_row['month_sin'] = np.sin(2 * np.pi * month / 12)
    last_row['month_cos'] = np.cos(2 * np.pi * month / 12)
    last_row['day_sin'] = np.sin(2 * np.pi * day / 31)
    last_row['day_cos'] = np.cos(2 * np.pi * day / 31)
    last_row['trend'] = len(df_prepared) + 1
    
    # Dự đoán từ tất cả mô hình
    predictions = {}
    
    for model_name in ['GradientBoosting', 'RandomForest', 'XGBoost']:
        try:
            # Load classifier
            classifier_path = os.path.join(models_dir, f"classifier_{model_name.lower()}.pkl")
            with open(classifier_path, 'rb') as f:
                classifier_pkg = pickle.load(f)
            classifier = classifier_pkg['model']
            scaler_c = classifier_pkg['scaler']
            feature_cols = classifier_pkg['features']
            metrics = classifier_pkg.get('metrics', {})
            
            # Load regressor
            regressor_path = os.path.join(models_dir, f"regressor_{model_name.lower()}.pkl")
            with open(regressor_path, 'rb') as f:
                regressor_pkg = pickle.load(f)
            regressor = regressor_pkg['model']
            scaler_r = regressor_pkg['scaler']
            metrics_r = regressor_pkg.get('metrics', {})
            
            # Lấy features
            X_pred = last_row[feature_cols].values.reshape(1, -1)
            X_pred_scaled = scaler_c.transform(X_pred)
            
            # Dự đoán
            has_rain_pred = classifier.predict(X_pred_scaled)[0]
            rain_prob = classifier.predict_proba(X_pred_scaled)[0][1]
            
            if has_rain_pred == 1:
                X_pred_r_scaled = scaler_r.transform(X_pred)
                rainfall_pred = regressor.predict(X_pred_r_scaled)[0]
                rainfall_pred = max(rainfall_pred, 0)
            else:
                rainfall_pred = 0
            
            # Get metrics from regressor (for MAE, RMSE, R²)
            mae = metrics_r.get('mae', 0)
            rmse = metrics_r.get('rmse', 0)
            r2_score = metrics_r.get('r2_score', 0)
            
            predictions[model_name] = {
                'has_rain': bool(has_rain_pred),
                'rain_probability': float(rain_prob),
                'predicted_rainfall': float(rainfall_pred),
                'mae': float(mae) if mae is not None else 0,
                'rmse': float(rmse) if rmse is not None else 0,
                'r2_score': float(r2_score) if r2_score is not None else 0
            }
        except Exception as e:
            # Fallback: return with 0 metrics
            predictions[model_name] = {
                'has_rain': False,
                'rain_probability': 0.0,
                'predicted_rainfall': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'r2_score': 0.0,
                'error': str(e)
            }
    
    return predictions

if __name__ == "__main__":
    result = predict_with_all_models('data/daily_combined.csv', 2024, 1, 15, 'models')
    for model_name, pred in result.items():
        print(f"\n{model_name}:")
        print(f"  Có mưa: {'Có' if pred['has_rain'] else 'Không'}")
        print(f"  Xác suất: {pred['rain_probability']*100:.1f}%")
        print(f"  Lượng mưa: {pred['predicted_rainfall']:.2f}mm")
