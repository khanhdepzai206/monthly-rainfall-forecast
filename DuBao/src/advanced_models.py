import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not installed. LSTM models will be skipped.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️  statsmodels not installed. ARIMA models will be skipped.")

def prepare_lstm_data(df, lookback=12):
    """
    Chuẩn bị dữ liệu cho LSTM
    lookback: số tháng trước để dự đoán tháng tiếp theo
    """
    data = df['rainfall'].values.reshape(-1, 1)
    
    # Normalize dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Tạo sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Train/test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape cho LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return (X_train, y_train, X_test, y_test, scaler)

def train_lstm_model(csv_path, lookback=12, epochs=100, model_path="models/lstm_model.h5"):
    """
    Train LSTM model cho dự báo chuỗi thời gian
    """
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Cannot train LSTM.")
        return None
    
    print("=" * 60)
    print("🤖 TRAINING LSTM MODEL")
    print("=" * 60)
    
    # Load dữ liệu
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 Preparing data (lookback={lookback} months)...")
    X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(df, lookback=lookback)
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set: {X_test.shape[0]} samples")
    
    # Build LSTM model
    print(f"\n🏗️  Building LSTM model...")
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    print("\n🎓 Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    # Predictions
    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_train_pred = scaler.inverse_transform(y_train_pred).flatten()
    y_test_pred = scaler.inverse_transform(y_test_pred).flatten()
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "=" * 60)
    print("📈 LSTM MODEL RESULTS")
    print("=" * 60)
    
    print("\n🎯 TRAINING SET METRICS:")
    print(f"   MAE : {train_mae:.4f} mm")
    print(f"   RMSE: {train_rmse:.4f} mm")
    print(f"   R²  : {train_r2:.4f}")
    
    print("\n🧪 TESTING SET METRICS:")
    print(f"   MAE : {test_mae:.4f} mm")
    print(f"   RMSE: {test_rmse:.4f} mm")
    print(f"   R²  : {test_r2:.4f}")
    
    # Save model
    model.save(model_path)
    print(f"\n✅ LSTM model saved: {model_path}")
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        },
        'lookback': lookback
    }

def train_arima_model(csv_path, order=(1, 1, 1), model_path="models/arima_model.pkl"):
    """
    Train ARIMA model cho dự báo chuỗi thời gian
    order: (p, d, q) - AutoRegressive, Integrated, Moving Average
    """
    if not STATSMODELS_AVAILABLE:
        print("❌ statsmodels not available. Cannot train ARIMA.")
        return None
    
    print("=" * 60)
    print("🤖 TRAINING ARIMA MODEL")
    print("=" * 60)
    
    # Load dữ liệu
    df = pd.read_csv(csv_path)
    
    # Train/test split
    train_size = int(len(df) * 0.8)
    train_data = df['rainfall'][:train_size]
    test_data = df['rainfall'][train_size:]
    
    print(f"\n📊 Data split:")
    print(f"   Training set: {len(train_data)} months")
    print(f"   Testing set: {len(test_data)} months")
    
    print(f"\n🏗️  Fitting ARIMA{order}...")
    
    # Fit ARIMA
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    
    print(f"\n📋 ARIMA Summary:")
    print(fitted_model.summary())
    
    # Predictions
    y_pred_train = fitted_model.fittedvalues
    y_pred_test = fitted_model.get_forecast(steps=len(test_data)).predicted_mean
    
    # Calculate metrics
    train_mae = mean_absolute_error(train_data[1:], y_pred_train[1:])  # Skip first point
    test_mae = mean_absolute_error(test_data, y_pred_test)
    
    train_rmse = np.sqrt(mean_squared_error(train_data[1:], y_pred_train[1:]))
    test_rmse = np.sqrt(mean_squared_error(test_data, y_pred_test))
    
    train_r2 = r2_score(train_data[1:], y_pred_train[1:])
    test_r2 = r2_score(test_data, y_pred_test)
    
    print("\n" + "=" * 60)
    print("📈 ARIMA MODEL RESULTS")
    print("=" * 60)
    
    print("\n🎯 TRAINING SET METRICS:")
    print(f"   MAE : {train_mae:.4f} mm")
    print(f"   RMSE: {train_rmse:.4f} mm")
    print(f"   R²  : {train_r2:.4f}")
    
    print("\n🧪 TESTING SET METRICS:")
    print(f"   MAE : {test_mae:.4f} mm")
    print(f"   RMSE: {test_rmse:.4f} mm")
    print(f"   R²  : {test_r2:.4f}")
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(fitted_model, f)
    print(f"\n✅ ARIMA model saved: {model_path}")
    
    return {
        'model': fitted_model,
        'order': order,
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }

def compare_models(csv_path):
    """
    So sánh hiệu suất của các mô hình
    """
    print("\n" + "=" * 80)
    print("🏆 MODEL COMPARISON")
    print("=" * 80)
    
    results = {}
    
    # Random Forest (already trained)
    from src.evaluate import evaluate_model
    print("\n1️⃣  Random Forest Model")
    print("-" * 80)
    try:
        rf_metrics = evaluate_model(csv_path, "models/rainfall_model_comparison.pkl")
        results['Random Forest'] = rf_metrics
    except Exception as e:
        print(f"❌ Error training Random Forest: {e}")
    
    # LSTM
    print("\n2️⃣  LSTM Model")
    print("-" * 80)
    try:
        lstm_result = train_lstm_model(csv_path)
        if lstm_result:
            results['LSTM'] = lstm_result['metrics']
    except Exception as e:
        print(f"❌ Error training LSTM: {e}")
    
    # ARIMA
    print("\n3️⃣  ARIMA Model")
    print("-" * 80)
    try:
        arima_result = train_arima_model(csv_path)
        if arima_result:
            results['ARIMA'] = arima_result['metrics']
    except Exception as e:
        print(f"❌ Error training ARIMA: {e}")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("📊 FINAL COMPARISON - TEST SET RESULTS")
    print("=" * 80)
    
    comparison_df = pd.DataFrame({
        model: {
            'MAE (mm)': metrics.get('test_mae', metrics.get('test_mae', 'N/A')),
            'RMSE (mm)': metrics.get('test_rmse', metrics.get('test_rmse', 'N/A')),
            'R²': metrics.get('test_r2', metrics.get('test_r2', 'N/A'))
        }
        for model, metrics in results.items()
    }).T
    
    print("\n" + comparison_df.to_string())
    
    # Best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1].get('test_r2', -float('inf')))
        print(f"\n🏆 Best Model: {best_model[0]}")
    
    print("\n" + "=" * 80)
    
    return results

if __name__ == "__main__":
    print("\n🔍 TRAINING ADVANCED MODELS\n")
    
    # Compare all models
    compare_models("data/monthly_rainfall.csv")
    
    print("\n✅ Advanced model training complete!")
