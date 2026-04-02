"""
Daily Rainfall Prediction Pipeline
Functions for Django web interface integration
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def get_daily_predictions():
    """
    Load trained models and predict rainfall for tomorrow

    Returns:
        tuple: (rf_prediction, lr_prediction, xgb_prediction)
    """
    try:
        # Load models
        models_dir = os.path.join(current_dir, '..', 'models')
        rf_model_path = os.path.join(models_dir, 'rf_daily_model.pkl')
        lr_model_path = os.path.join(models_dir, 'lr_daily_model.pkl')
        xgb_model_path = os.path.join(models_dir, 'xgb_daily_model.pkl')

        if not all(os.path.exists(p) for p in [rf_model_path, lr_model_path, xgb_model_path]):
            raise FileNotFoundError("Không tìm thấy các file model đã train")

        rf_model = joblib.load(rf_model_path)
        lr_model = joblib.load(lr_model_path)
        xgb_model = joblib.load(xgb_model_path)

        # Load latest features for tomorrow prediction
        data_dir = os.path.join(current_dir, '..', 'data')
        features_file = os.path.join(data_dir, 'daily_features.csv')

        if not os.path.exists(features_file):
            raise FileNotFoundError("Không tìm thấy file daily_features.csv")

        df = pd.read_csv(features_file)

        # Get latest row for prediction (tomorrow's features)
        latest_features = df.iloc[-1:].copy()

        # Remove target column if exists
        if 'rainfall' in latest_features.columns:
            latest_features = latest_features.drop('rainfall', axis=1)

        # Ensure all required features are present (35 lag features)
        required_features = [
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

        missing_features = [f for f in required_features if f not in latest_features.columns]
        if missing_features:
            raise ValueError(f"Thiếu các features: {missing_features}")

        # Select only required features in correct order
        features_for_prediction = latest_features[required_features]

        # Make predictions
        rf_pred = float(rf_model.predict(features_for_prediction)[0])
        lr_pred = float(lr_model.predict(features_for_prediction)[0])
        xgb_pred = float(xgb_model.predict(features_for_prediction)[0])

        return rf_pred, lr_pred, xgb_pred

    except Exception as e:
        print(f"Lỗi trong get_daily_predictions: {str(e)}")
        raise

def retrain_models():
    """
    Retrain all models with updated data including latest actual rainfall
    """
    try:
        print("Bắt đầu retrain models...")

        # Import training function
        from train_daily_models import train_daily_models

        # Run training
        train_daily_models()

        print("Hoàn thành retrain models!")

    except Exception as e:
        print(f"Lỗi trong retrain_models: {str(e)}")
        raise

def get_best_model(predictions):
    """
    Determine best model based on historical performance

    Args:
        predictions: tuple of (rf_pred, lr_pred, xgb_pred)

    Returns:
        str: 'RF', 'LR', or 'XGB'
    """
    try:
        # Load model metrics
        models_dir = os.path.join(current_dir, '..', 'models')
        metrics_file = os.path.join(models_dir, 'model_metrics.json')

        if os.path.exists(metrics_file):
            import json
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Get MAE scores (lower is better)
            rf_mae = metrics.get('rf_daily', {}).get('mae', float('inf'))
            lr_mae = metrics.get('lr_daily', {}).get('mae', float('inf'))
            xgb_mae = metrics.get('xgb_daily', {}).get('mae', float('inf'))

            # Find model with lowest MAE
            mae_scores = {'RF': rf_mae, 'LR': lr_mae, 'XGB': xgb_mae}
            best_model = min(mae_scores, key=mae_scores.get)

            return best_model
        else:
            # Default to RF if no metrics available
            return 'RF'

    except Exception as e:
        print(f"Lỗi trong get_best_model: {str(e)}")
        return 'RF'

if __name__ == "__main__":
    # Test functions
    try:
        print("Testing get_daily_predictions...")
        rf_pred, lr_pred, xgb_pred = get_daily_predictions()
        print(f"Predictions - RF: {rf_pred:.2f}, LR: {lr_pred:.2f}, XGB: {xgb_pred:.2f}")

        best = get_best_model((rf_pred, lr_pred, xgb_pred))
        print(f"Best model: {best}")

    except Exception as e:
        print(f"Test failed: {str(e)}")