import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    """
    Tạo thêm features để cải thiện mô hình
    """
    df = df.sort_values(['year', 'month']).reset_index(drop=True)
    
    # Seasonal features (Sin/Cos encoding)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features (lượng mưa từ tháng trước)
    df['rainfall_lag_1'] = df['rainfall'].shift(1)
    df['rainfall_lag_3'] = df['rainfall'].shift(3)
    df['rainfall_lag_12'] = df['rainfall'].shift(12)  # Năm trước
    
    # Moving average
    df['rainfall_ma_3'] = df['rainfall'].rolling(window=3, min_periods=1).mean()
    df['rainfall_ma_12'] = df['rainfall'].rolling(window=12, min_periods=1).mean()
    
    # Trend
    df['trend'] = range(len(df))
    
    # Quarter
    df['quarter'] = (df['month'] - 1) // 3 + 1
    
    # Drop NaN rows
    df = df.dropna()
    
    return df

def train_improved_rf_model(csv_path, model_path="models/rainfall_model_improved.pkl", scaler_path="models/scaler.pkl"):
    """
    Train Random Forest với features cải thiện
    """
    print("=" * 70)
    print("🤖 TRAINING IMPROVED RANDOM FOREST MODEL")
    print("=" * 70)
    
    # Load dữ liệu
    df = pd.read_csv(csv_path)
    
    # Tạo features
    print("\n📊 Creating advanced features...")
    df = create_features(df)
    
    print(f"✓ Dataset shape after feature engineering: {df.shape}")
    print(f"✓ Features: {list(df.columns[df.columns != 'rainfall'])}")
    
    # Tách input và output
    feature_cols = [col for col in df.columns if col not in ['rainfall', 'year', 'month']]
    feature_cols.extend(['year', 'month'])
    
    X = df[feature_cols]
    y = df['rainfall']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=False  # Time series - không shuffle
    )
    
    print(f"\n📈 Dataset Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train Random Forest dengan hyperparameters tốt hơn
    print(f"\n🏗️  Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=500,           # Tăng từ 200
        max_depth=20,               # Thêm constraint
        min_samples_split=5,        # Giảm để bắt được pattern
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,                  # Dùng multi-processing
        max_features='sqrt'         # Reduce overfitting
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "=" * 70)
    print("📊 IMPROVED RANDOM FOREST RESULTS")
    print("=" * 70)
    
    print("\n🎯 TRAINING SET:")
    print(f"   MAE : {train_mae:.4f} mm")
    print(f"   RMSE: {train_rmse:.4f} mm")
    print(f"   R²  : {train_r2:.4f}")
    
    print("\n🧪 TESTING SET:")
    print(f"   MAE : {test_mae:.4f} mm")
    print(f"   RMSE: {test_rmse:.4f} mm")
    print(f"   R²  : {test_r2:.4f}")
    
    # Feature importance
    print("\n🔍 FEATURE IMPORTANCE (Top 10):")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    # Save models
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"\n✅ Models saved:")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }

def train_gradient_boosting_model(csv_path, model_path="models/rainfall_model_gb.pkl"):
    """
    Train Gradient Boosting model (thường tốt hơn Random Forest)
    """
    print("\n" + "=" * 70)
    print("🤖 TRAINING GRADIENT BOOSTING MODEL")
    print("=" * 70)
    
    # Load dữ liệu
    df = pd.read_csv(csv_path)
    df = create_features(df)
    
    # Tách features
    feature_cols = [col for col in df.columns if col not in ['rainfall', 'year', 'month']]
    feature_cols.extend(['year', 'month'])
    
    X = df[feature_cols]
    y = df['rainfall']
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Train Gradient Boosting
    print("\n🏗️  Training Gradient Boosting...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        subsample=0.8
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "=" * 70)
    print("📊 GRADIENT BOOSTING RESULTS")
    print("=" * 70)
    
    print("\n🎯 TRAINING SET:")
    print(f"   MAE : {train_mae:.4f} mm")
    print(f"   RMSE: {train_rmse:.4f} mm")
    print(f"   R²  : {train_r2:.4f}")
    
    print("\n🧪 TESTING SET:")
    print(f"   MAE : {test_mae:.4f} mm")
    print(f"   RMSE: {test_rmse:.4f} mm")
    print(f"   R²  : {test_r2:.4f}")
    
    # Feature importance
    print("\n🔍 FEATURE IMPORTANCE (Top 10):")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\n✅ Model saved: {model_path}")
    
    return {
        'model': model,
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }

if __name__ == "__main__":
    print("\n🔄 TRAINING IMPROVED MODELS\n")
    
    # Train improved Random Forest
    rf_result = train_improved_rf_model("data/monthly_rainfall.csv")
    
    # Train Gradient Boosting
    gb_result = train_gradient_boosting_model("data/monthly_rainfall.csv")
    
    # Compare
    print("\n" + "=" * 70)
    print("🏆 MODEL COMPARISON (IMPROVED)")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        'Improved RF': {
            'MAE': rf_result['metrics']['test_mae'],
            'RMSE': rf_result['metrics']['test_rmse'],
            'R²': rf_result['metrics']['test_r2']
        },
        'Gradient Boosting': {
            'MAE': gb_result['metrics']['test_mae'],
            'RMSE': gb_result['metrics']['test_rmse'],
            'R²': gb_result['metrics']['test_r2']
        }
    }).T
    
    print("\n" + comparison.to_string())
    
    best_model = 'Gradient Boosting' if gb_result['metrics']['test_r2'] > rf_result['metrics']['test_r2'] else 'Improved RF'
    print(f"\n🏆 Best Model: {best_model}")
    print("\n✅ Training complete!")
