import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_AVAILABLE = True
except ImportError:
    SARIMA_AVAILABLE = False
    print("⚠️  statsmodels not installed. Run: pip install statsmodels")

def train_sarima_model(csv_path, model_path="models/sarima_model.pkl", order=(1,1,1), seasonal_order=(1,1,1,12)):
    """
    Train SARIMA model cho dự báo lượng mưa tháng
    
    SARIMA(p,d,q)(P,D,Q,s):
    - p,d,q: Non-seasonal parameters
    - P,D,Q,s: Seasonal parameters (s=12 for monthly data)
    """
    if not SARIMA_AVAILABLE:
        print("❌ statsmodels not available. Cannot train SARIMA.")
        return None
    
    print("=" * 70)
    print("🤖 TRAINING SARIMA MODEL")
    print("=" * 70)
    
    # Load dữ liệu
    df = pd.read_csv(csv_path)
    rainfall_series = df['rainfall'].values
    
    print(f"\n📊 Data Shape: {len(rainfall_series)} months")
    print(f"📊 Date range: {df['year'].min()}-{df['month'].min()} to {df['year'].max()}-{df['month'].max()}")
    
    # Train/Test split (80/20)
    train_size = int(len(rainfall_series) * 0.8)
    train_data = rainfall_series[:train_size]
    test_data = rainfall_series[train_size:]
    
    print(f"\n📈 Dataset Split:")
    print(f"   Training: {len(train_data)} months")
    print(f"   Testing: {len(test_data)} months")
    
    # Fit SARIMA
    print(f"\n🏗️  Fitting SARIMA{order}{seasonal_order}...")
    print("   (Điều này có thể mất vài phút...)")
    
    try:
        model = SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        results = model.fit(disp=False, maxiter=1000)
        
        # Predictions
        y_pred_train = results.fittedvalues
        y_pred_test = results.get_forecast(steps=len(test_data)).predicted_mean.values
        
        # Metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Train metrics (bỏ NaN đầu tiên)
        valid_train_idx = ~np.isnan(y_pred_train)
        train_mae = mean_absolute_error(train_data[valid_train_idx], y_pred_train[valid_train_idx])
        train_rmse = np.sqrt(mean_squared_error(train_data[valid_train_idx], y_pred_train[valid_train_idx]))
        train_r2 = r2_score(train_data[valid_train_idx], y_pred_train[valid_train_idx])
        
        # Test metrics
        test_mae = mean_absolute_error(test_data, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(test_data, y_pred_test))
        test_r2 = r2_score(test_data, y_pred_test)
        
        print("\n" + "=" * 70)
        print("📈 SARIMA MODEL RESULTS")
        print("=" * 70)
        
        print("\n🎯 TRAINING SET:")
        print(f"   MAE : {train_mae:.4f} mm")
        print(f"   RMSE: {train_rmse:.4f} mm")
        print(f"   R²  : {train_r2:.4f}")
        
        print("\n🧪 TESTING SET:")
        print(f"   MAE : {test_mae:.4f} mm")
        print(f"   RMSE: {test_rmse:.4f} mm")
        print(f"   R²  : {test_r2:.4f}")
        
        # Model summary
        print("\n📋 SARIMA Summary:")
        print(results.summary())
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n✅ Model saved: {model_path}")
        
        return {
            'model': results,
            'order': order,
            'seasonal_order': seasonal_order,
            'metrics': {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
        }
        
    except Exception as e:
        print(f"\n❌ Error fitting SARIMA: {e}")
        print("   Cố gắng với default parameters...")
        return None

def find_best_sarima(csv_path):
    """
    Tìm tham số SARIMA tốt nhất bằng auto_arima
    """
    if not SARIMA_AVAILABLE:
        return None
    
    try:
        from pmdarima import auto_arima
        
        print("=" * 70)
        print("🔍 FINDING BEST SARIMA PARAMETERS")
        print("=" * 70)
        
        df = pd.read_csv(csv_path)
        rainfall_series = df['rainfall'].values
        
        print("\n🏗️  Running auto_arima (có thể mất 5-10 phút)...")
        
        auto_model = auto_arima(
            rainfall_series,
            seasonal=True,
            m=12,  # Monthly seasonality
            start_p=0, start_q=0, start_P=0, start_Q=0,
            max_p=5, max_q=5, max_P=2, max_Q=2,
            max_d=2, max_D=1,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        print(f"\n✅ Best parameters found:")
        print(f"   Order: {auto_model.order}")
        print(f"   Seasonal Order: {auto_model.seasonal_order}")
        
        return auto_model.order, auto_model.seasonal_order
        
    except ImportError:
        print("⚠️  pmdarima not installed. Run: pip install pmdarima")
        print("   Using default SARIMA(1,1,1)(1,1,1,12)")
        return (1,1,1), (1,1,1,12)

if __name__ == "__main__":
    print("\n🔄 TRAINING SARIMA MODEL\n")
    
    # Tìm tham số tốt nhất (optional, mất thời gian)
    # order, seasonal_order = find_best_sarima("data/monthly_rainfall.csv")
    
    # Hoặc dùng tham số mặc định
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    
    # Train model
    result = train_sarima_model(
        "data/monthly_rainfall.csv",
        order=order,
        seasonal_order=seasonal_order
    )
    
    print("\n✅ Training complete!")
