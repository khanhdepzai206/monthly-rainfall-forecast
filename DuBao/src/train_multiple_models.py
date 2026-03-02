"""
Train multiple models (GradientBoosting, RandomForest, XGBoost)
Lưu tất cả để có thể so sánh
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

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
    df['rainfall_std_7'] = df['rainfall'].rolling(window=7, min_periods=1).std()
    
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
    
    return df

def train_multiple_models(csv_path, output_dir="models"):
    """Train GradientBoosting, RandomForest, XGBoost"""
    
    print("📊 Đang load dữ liệu...")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} records")
    rain_count = (df['rainfall'] > 0).sum()
    print(f"  - Có mưa: {rain_count} ({rain_count/len(df)*100:.1f}%)")
    print(f"  - Không mưa: {len(df) - rain_count} ({(len(df)-rain_count)/len(df)*100:.1f}%)")
    
    # Chuẩn bị features
    df_prepared = prepare_daily_features(df.copy())
    df_prepared = df_prepared.dropna()
    
    # Target
    y_classifier = (df_prepared['rainfall'] > 0).astype(int)
    y_regressor = df_prepared.loc[y_classifier == 1, 'rainfall']
    
    # Features
    feature_cols = [col for col in df_prepared.columns if col not in ['date', 'rainfall', 'year', 'month', 'day']]
    X = df_prepared[feature_cols]
    X_rain = df_prepared[y_classifier == 1][feature_cols]
    
    # Split
    X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_classifier, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_rain, y_regressor, test_size=0.2, random_state=42)
    
    # Scale
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train)
    X_test_c_scaled = scaler_c.transform(X_test)
    
    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)
    
    # ===== TRAIN CLASSIFIERS =====
    print("\n🎯 Huấn luyện Classifiers...")
    
    classifiers = {
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42, verbosity=0)
    }
    
    classifier_results = {}
    
    for name, model in classifiers.items():
        print(f"\n  {name}...")
        model.fit(X_train_c_scaled, y_train_c)
        y_pred = model.predict(X_test_c_scaled)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(y_test_c, y_pred)
        prec = precision_score(y_test_c, y_pred)
        rec = recall_score(y_test_c, y_pred)
        f1 = f1_score(y_test_c, y_pred)
        
        classifier_results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
        
        print(f"    Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Save
        with open(f"{output_dir}/classifier_{name.lower()}.pkl", 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler_c,
                'features': feature_cols
            }, f)
    
    # ===== TRAIN REGRESSORS =====
    print("\n🌧️ Huấn luyện Regressors...")
    
    regressors = {
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, verbosity=0)
    }
    
    regressor_results = {}
    
    for name, model in regressors.items():
        print(f"\n  {name}...")
        model.fit(X_train_r_scaled, y_train_r)
        y_pred_r = model.predict(X_test_r_scaled)
        
        mae = mean_absolute_error(y_test_r, y_pred_r)
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
        r2 = r2_score(y_test_r, y_pred_r)
        mape = np.mean(np.abs((y_test_r - y_pred_r) / y_test_r)) * 100
        
        regressor_results[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        print(f"    MAE: {mae:.2f}mm, RMSE: {rmse:.2f}mm, R²: {r2:.4f}, MAPE: {mape:.2f}%")
        
        # Save
        with open(f"{output_dir}/regressor_{name.lower()}.pkl", 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler_r,
                'features': feature_cols
            }, f)
    
    # Save results
    print("\n💾 Lưu kết quả so sánh...")
    with open(f"{output_dir}/comparison_results.pkl", 'wb') as f:
        pickle.dump({
            'classifier_results': classifier_results,
            'regressor_results': regressor_results,
            'test_count': len(X_test),
            'rain_count': (y_test_c == 1).sum()
        }, f)
    
    print(f"\n✅ Đã lưu tất cả mô hình vào {output_dir}/")
    
    # Print summary
    print("\n📊 TÓM TẮT CLASSIFIER:")
    best_f1 = max(r['f1'] for r in classifier_results.values())
    for name, metrics in classifier_results.items():
        star = "⭐" if metrics['f1'] == best_f1 else ""
        print(f"  {name:20s} F1: {metrics['f1']:.4f} {star}")
    
    print("\n📊 TÓM TẮT REGRESSOR:")
    best_r2 = max(r['r2'] for r in regressor_results.values())
    for name, metrics in regressor_results.items():
        star = "⭐" if metrics['r2'] == best_r2 else ""
        print(f"  {name:20s} R²: {metrics['r2']:.4f} {star}")

if __name__ == "__main__":
    train_multiple_models(
        csv_path="data/daily_combined.csv",
        output_dir="models"
    )
