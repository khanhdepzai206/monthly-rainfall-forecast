import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Thử import XGBoost nếu có
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost chưa cài. Cài: pip install xgboost")

from train_two_step_daily import prepare_daily_features

def train_compare_models(csv_path, output_dir="models"):
    """
    Train 2-3 mô hình khác nhau và so sánh kết quả
    """
    print("📊 Đang load dữ liệu ngày...")
    df = pd.read_csv(csv_path)
    
    # Chuẩn bị features
    df = prepare_daily_features(df)
    df = df.dropna()
    
    print(f"✓ Loaded {len(df)} ngày dữ liệu")
    print(f"  - Ngày có mưa: {(df['rainfall'] > 0).sum()} ({(df['rainfall'] > 0).sum() / len(df) * 100:.1f}%)")
    print(f"  - Ngày không mưa: {(df['rainfall'] == 0).sum()} ({(df['rainfall'] == 0).sum() / len(df) * 100:.1f}%)")
    
    # Định nghĩa target
    df['has_rain'] = (df['rainfall'] > 0).astype(int)
    
    # Chọn features
    feature_cols = [col for col in df.columns if col not in ['rainfall', 'has_rain', 'year', 'month', 'day', 'date']]
    X = df[feature_cols]
    
    # ===== CLASSIFIER MODELS =====
    print("\n" + "="*60)
    print("🎯 HUẤN LUYỆN CLASSIFIER MODELS")
    print("="*60)
    
    y_classifier = df['has_rain']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_classifier, test_size=0.2, random_state=42, stratify=y_classifier
    )
    
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)
    
    classifier_models = {
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=7,
            min_samples_split=10, min_samples_leaf=5, subsample=0.8, random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
    }
    
    if HAS_XGBOOST:
        classifier_models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=7,
            subsample=0.8, random_state=42, verbosity=0, n_jobs=-1
        )
    
    classifier_results = {}
    
    for model_name, model in classifier_models.items():
        print(f"\n🔄 Training {model_name}...")
        start_time = time.time()
        
        model.fit(X_train_c_scaled, y_train_c)
        
        train_time = time.time() - start_time
        y_pred = model.predict(X_test_c_scaled)
        y_pred_proba = model.predict_proba(X_test_c_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test_c, y_pred)
        precision = precision_score(y_test_c, y_pred)
        recall = recall_score(y_test_c, y_pred)
        f1 = f1_score(y_test_c, y_pred)
        
        classifier_results[model_name] = {
            'model': model,
            'scaler': scaler_c,
            'features': feature_cols,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': train_time,
            'y_pred': y_pred,
            'y_test': y_test_c
        }
        
        print(f"  ✓ Accuracy:  {accuracy:.4f}")
        print(f"  ✓ Precision: {precision:.4f}")
        print(f"  ✓ Recall:    {recall:.4f}")
        print(f"  ✓ F1-Score:  {f1:.4f}")
        print(f"  ⏱️  Train time: {train_time:.2f}s")
        
        # Lưu model
        model_path = f"{output_dir}/daily_classifier_{model_name.lower()}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler_c, 'features': feature_cols}, f)
        print(f"  💾 Saved: {model_path}")
    
    # ===== REGRESSOR MODELS =====
    print("\n" + "="*60)
    print("🌧️  HUẤN LUYỆN REGRESSOR MODELS")
    print("="*60)
    
    # Chỉ lấy ngày có mưa
    rain_mask = df['has_rain'] == 1
    X_rain = X[rain_mask]
    y_rain = df[rain_mask]['rainfall']
    
    print(f"✓ Dữ liệu huấn luyện Regressor: {len(X_rain)} ngày có mưa")
    
    if len(X_rain) > 100:
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_rain, y_rain, test_size=0.2, random_state=42
        )
        
        scaler_r = StandardScaler()
        X_train_r_scaled = scaler_r.fit_transform(X_train_r)
        X_test_r_scaled = scaler_r.transform(X_test_r)
        
        regressor_models = {
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=7,
                min_samples_split=10, min_samples_leaf=5, subsample=0.8, random_state=42
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            )
        }
        
        if HAS_XGBOOST:
            regressor_models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=7,
                subsample=0.8, random_state=42, verbosity=0, n_jobs=-1
            )
        
        regressor_results = {}
        
        for model_name, model in regressor_models.items():
            print(f"\n🔄 Training {model_name}...")
            start_time = time.time()
            
            model.fit(X_train_r_scaled, y_train_r)
            
            train_time = time.time() - start_time
            y_pred = model.predict(X_test_r_scaled)
            y_pred = np.maximum(y_pred, 0)  # Không âm
            
            # Metrics
            mae = mean_absolute_error(y_test_r, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
            r2 = r2_score(y_test_r, y_pred)
            mape = np.mean(np.abs((y_test_r - y_pred) / y_test_r)) * 100
            
            regressor_results[model_name] = {
                'model': model,
                'scaler': scaler_r,
                'features': feature_cols,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'train_time': train_time,
                'y_pred': y_pred,
                'y_test': y_test_r.values
            }
            
            print(f"  ✓ MAE:  {mae:.2f} mm")
            print(f"  ✓ RMSE: {rmse:.2f} mm")
            print(f"  ✓ R²:   {r2:.4f}")
            print(f"  ✓ MAPE: {mape:.2f}%")
            print(f"  ⏱️  Train time: {train_time:.2f}s")
            
            # Lưu model
            model_path = f"{output_dir}/daily_regressor_{model_name.lower()}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({'model': model, 'scaler': scaler_r, 'features': feature_cols}, f)
            print(f"  💾 Saved: {model_path}")
    
    # ===== SO SÁNH KẾT QUẢ =====
    print("\n" + "="*60)
    print("📊 SO SÁNH CLASSIFIER MODELS")
    print("="*60)
    
    comparison_data_c = []
    for model_name, results in classifier_results.items():
        comparison_data_c.append({
            'Model': model_name,
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'F1-Score': f"{results['f1']:.4f}",
            'Train (s)': f"{results['train_time']:.2f}"
        })
    
    df_compare_c = pd.DataFrame(comparison_data_c)
    print("\n" + df_compare_c.to_string(index=False))
    
    # Tìm mô hình tốt nhất
    best_classifier = max(classifier_results.items(), key=lambda x: x[1]['f1'])
    print(f"\n🏆 Classifier tốt nhất: {best_classifier[0]} (F1: {best_classifier[1]['f1']:.4f})")
    
    if len(regressor_results) > 0:
        print("\n" + "="*60)
        print("📊 SO SÁNH REGRESSOR MODELS")
        print("="*60)
        
        comparison_data_r = []
        for model_name, results in regressor_results.items():
            comparison_data_r.append({
                'Model': model_name,
                'MAE (mm)': f"{results['mae']:.2f}",
                'RMSE (mm)': f"{results['rmse']:.2f}",
                'R²': f"{results['r2']:.4f}",
                'MAPE (%)': f"{results['mape']:.2f}",
                'Train (s)': f"{results['train_time']:.2f}"
            })
        
        df_compare_r = pd.DataFrame(comparison_data_r)
        print("\n" + df_compare_r.to_string(index=False))
        
        # Tìm mô hình tốt nhất
        best_regressor = max(regressor_results.items(), key=lambda x: x[1]['r2'])
        print(f"\n🏆 Regressor tốt nhất: {best_regressor[0]} (R²: {best_regressor[1]['r2']:.4f})")
        
        # Lưu kết quả so sánh
        print(f"\n💾 Lưu kết quả so sánh...")
        with open(f"{output_dir}/comparison_results.pkl", 'wb') as f:
            pickle.dump({
                'classifier_results': classifier_results,
                'regressor_results': regressor_results,
                'best_classifier': best_classifier[0],
                'best_regressor': best_regressor[0]
            }, f)
        print(f"✅ Đã lưu comparison_results.pkl")

if __name__ == "__main__":
    train_compare_models(
        csv_path="data/daily_combined.csv",
        output_dir="models"
    )
