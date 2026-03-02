"""
Script: Đánh giá độ chính xác của các mô hình hiện có
Kiểm tra performance trên test set
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

def load_model(path):
    """Load pickle model"""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def evaluate_models():
    """Đánh giá độ chính xác của các mô hình hiện có"""
    
    print("\n" + "="*70)
    print("📊 ĐÁNH GIÁ ĐỘ CHÍNH XÁC CỦA CÁC MÔ HÌNH")
    print("="*70)
    
    models_dir = "models"
    data_dir = "data"
    
    # Load dữ liệu test
    print("\n📥 Load dữ liệu test...")
    df = pd.read_csv(os.path.join(data_dir, "daily_combined.csv"))
    
    from train_two_step_daily import prepare_daily_features
    df = prepare_daily_features(df)
    df = df.dropna()
    
    # Chia train/test (80/20)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    print(f"  Train set: {len(df_train)} mẫu")
    print(f"  Test set: {len(df_test)} mẫu")
    
    # Tạo target
    df_test['has_rain'] = (df_test['rainfall'] > 0).astype(int)
    
    # Features
    feature_cols = [col for col in df_test.columns 
                   if col not in ['rainfall', 'has_rain', 'year', 'month', 'day', 'date']]
    X_test = df_test[feature_cols]
    y_test_rain = df_test['has_rain']
    y_test_amount = df_test['rainfall']
    
    # ===== CLASSIFIER MODELS =====
    print("\n" + "-"*70)
    print("🎯 CLASSIFIER MODELS (Dự đoán có mưa hay không)")
    print("-"*70)
    
    classifier_results = {}
    
    classifier_files = {
        'GradientBoosting': 'daily_gradient_boosting_model.pkl',
        'RandomForest': 'daily_random_forest_model.pkl',
        'XGBoost': 'daily_xgboost_model.pkl',
        'XGBoost_Tuned': 'daily_xgb_tuned_model.pkl',
        'Ensemble': 'daily_ensemble_model.pkl',
        'Stacking': 'daily_stacking_model.pkl'
    }
    
    for model_name, filename in classifier_files.items():
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"\n⚠️ {model_name}: File không tìm thấy ({filename})")
            continue
        
        print(f"\n🔄 Testing {model_name}...")
        
        try:
            model_pkg = load_model(filepath)
            
            if isinstance(model_pkg, dict) and 'model' in model_pkg:
                model = model_pkg['model']
                scaler = model_pkg.get('scaler')
            else:
                model = model_pkg
                scaler = None
            
            # Chuẩn bị dữ liệu
            X_test_proc = X_test.copy()
            
            if scaler:
                X_test_proc = scaler.transform(X_test)
            
            # Dự đoán
            y_pred = model.predict(X_test_proc)
            
            # Nếu là số thực, chuyển thành binary
            if y_pred.dtype == 'float':
                y_pred = (y_pred > 0.5).astype(int)
            
            # Metrics
            acc = accuracy_score(y_test_rain, y_pred)
            prec = precision_score(y_test_rain, y_pred, zero_division=0)
            rec = recall_score(y_test_rain, y_pred, zero_division=0)
            f1 = f1_score(y_test_rain, y_pred, zero_division=0)
            
            classifier_results[model_name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
            
            print(f"  ✅ Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
            print(f"  ✅ Precision: {prec:.4f}")
            print(f"  ✅ Recall:    {rec:.4f}")
            print(f"  ✅ F1-Score:  {f1:.4f}")
            
        except Exception as e:
            print(f"  ❌ Lỗi: {str(e)}")
    
    # ===== REGRESSOR MODELS =====
    print("\n" + "-"*70)
    print("📈 REGRESSOR MODELS (Dự đoán lượng mưa)")
    print("-"*70)
    
    regressor_results = {}
    
    regressor_files = {
        'GradientBoosting': 'daily_gradient_boosting_model.pkl',
        'RandomForest': 'daily_random_forest_model.pkl',
        'XGBoost': 'daily_xgboost_model.pkl',
        'XGBoost_Tuned': 'daily_xgb_tuned_model.pkl',
    }
    
    # Chỉ test trên ngày có mưa
    rain_mask = y_test_amount > 0
    X_test_rain = X_test[rain_mask]
    y_test_amount_rain = y_test_amount[rain_mask]
    
    print(f"\nTest set (chỉ ngày có mưa): {len(X_test_rain)} mẫu")
    
    for model_name, filename in regressor_files.items():
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath) or len(X_test_rain) == 0:
            continue
        
        print(f"\n🔄 Testing {model_name}...")
        
        try:
            model_pkg = load_model(filepath)
            
            if isinstance(model_pkg, dict) and 'model' in model_pkg:
                model = model_pkg['model']
                scaler = model_pkg.get('scaler')
            else:
                model = model_pkg
                scaler = None
            
            # Chuẩn bị dữ liệu
            X_test_proc = X_test_rain.copy()
            
            if scaler:
                X_test_proc = scaler.transform(X_test_rain)
            
            # Dự đoán
            y_pred = model.predict(X_test_proc)
            y_pred = np.maximum(y_pred, 0)  # Không âm
            
            # Metrics
            mae = mean_absolute_error(y_test_amount_rain, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_amount_rain, y_pred))
            r2 = r2_score(y_test_amount_rain, y_pred)
            mape = np.mean(np.abs((y_test_amount_rain - y_pred) / y_test_amount_rain)) * 100
            
            regressor_results[model_name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            }
            
            print(f"  ✅ MAE:  {mae:.2f} mm")
            print(f"  ✅ RMSE: {rmse:.2f} mm")
            print(f"  ✅ R²:   {r2:.4f}")
            print(f"  ✅ MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"  ❌ Lỗi: {str(e)}")
    
    # ===== SO SÁNH KẾT QUẢ =====
    if classifier_results:
        print("\n" + "="*70)
        print("📊 SO SÁNH CLASSIFIER MODELS")
        print("="*70)
        
        df_class = pd.DataFrame(classifier_results).T
        df_class = df_class.sort_values('f1', ascending=False)
        
        print("\n" + df_class.to_string())
        
        best_classifier = df_class.index[0]
        print(f"\n🏆 Classifier tốt nhất: {best_classifier} (F1: {df_class.loc[best_classifier, 'f1']:.4f})")
    
    if regressor_results:
        print("\n" + "="*70)
        print("📊 SO SÁNH REGRESSOR MODELS")
        print("="*70)
        
        df_reg = pd.DataFrame(regressor_results).T
        df_reg = df_reg.sort_values('r2', ascending=False)
        
        print("\n" + df_reg.to_string())
        
        best_regressor = df_reg.index[0]
        print(f"\n🏆 Regressor tốt nhất: {best_regressor} (R²: {df_reg.loc[best_regressor, 'r2']:.4f})")
    
    # Lưu kết quả
    print("\n" + "="*70)
    print("💾 Lưu kết quả đánh giá...")
    
    results = {
        'classifier_results': classifier_results,
        'regressor_results': regressor_results,
        'test_set_size': len(df_test),
        'rain_days': rain_mask.sum()
    }
    
    with open(os.path.join(models_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    with open(os.path.join(models_dir, 'model_metrics.json'), 'w', encoding='utf-8') as f:
        import json
        
        # Chuyển dict để JSON serializable
        json_results = {
            'classifier': {k: {mk: float(mv) for mk, mv in v.items()} 
                          for k, v in classifier_results.items()},
            'regressor': {k: {mk: float(mv) for mk, mv in v.items()} 
                         for k, v in regressor_results.items()},
            'test_set_size': int(len(df_test)),
            'rain_days': int(rain_mask.sum())
        }
        json.dump(json_results, f, indent=2)
    
    print(f"✅ Đã lưu evaluation_results.pkl")
    print(f"✅ Đã lưu model_metrics.json")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    evaluate_models()
