import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate_model(csv_path, model_path, test_size=0.2, random_state=42):
    """
    Đánh giá mô hình với train/test split
    
    Args:
        csv_path: Đường dẫn file dữ liệu monthly
        model_path: Đường dẫn lưu mô hình
        test_size: Tỷ lệ test set (default 20%)
        random_state: Random seed
    
    Returns:
        dict: Chứa metrics (MAE, RMSE, R², predictions)
    """
    
    # Load dữ liệu
    df = pd.read_csv(csv_path)
    
    # Tách input và output
    X = df[["year", "month"]]
    y = df["rainfall"]
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"📊 Dataset Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Test size: {test_size*100}%\n")
    
    # Tạo và train mô hình
    model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Dự đoán
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Tính metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # In kết quả
    print("=" * 50)
    print("📈 MODEL EVALUATION RESULTS")
    print("=" * 50)
    
    print("\n🎯 TRAINING SET METRICS:")
    print(f"   MAE  (Mean Absolute Error)   : {train_mae:.4f} mm")
    print(f"   RMSE (Root Mean Squared Error): {train_rmse:.4f} mm")
    print(f"   R²   (Coefficient of Determination): {train_r2:.4f}")
    
    print("\n🧪 TESTING SET METRICS:")
    print(f"   MAE  (Mean Absolute Error)   : {test_mae:.4f} mm")
    print(f"   RMSE (Root Mean Squared Error): {test_rmse:.4f} mm")
    print(f"   R²   (Coefficient of Determination): {test_r2:.4f}")
    
    print("\n📊 MODEL QUALITY INTERPRETATION:")
    if test_r2 > 0.8:
        print(f"   ✅ Excellent model (R² > 0.8)")
    elif test_r2 > 0.6:
        print(f"   ✅ Good model (R² > 0.6)")
    elif test_r2 > 0.4:
        print(f"   ⚠️  Moderate model (R² > 0.4)")
    else:
        print(f"   ❌ Weak model (R² < 0.4)")
    
    print("\n" + "=" * 50)
    
    # Feature importance
    print("\n🔍 FEATURE IMPORTANCE:")
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"   {feature}: {importance:.4f}")
    
    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved: {model_path}")
    
    # Return metrics
    metrics = {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'X_test': X_test,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'model': model
    }
    
    return metrics

def plot_predictions(y_test, y_pred, output_file="predictions_plot.png"):
    """
    Vẽ biểu đồ so sánh giá trị thực vs dự đoán
    """
    plt.figure(figsize=(12, 6))
    
    # Plot thực tế vs dự đoán
    plt.scatter(y_test, y_pred, alpha=0.5, s=30)
    
    # Đường y=x (perfect prediction)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    plt.xlabel('Actual Rainfall (mm)', fontsize=12)
    plt.ylabel('Predicted Rainfall (mm)', fontsize=12)
    plt.title('Actual vs Predicted Rainfall', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"📊 Plot saved: {output_file}")
    plt.close()

def plot_residuals(y_test, y_pred, output_file="residuals_plot.png"):
    """
    Vẽ biểu đồ phần dư (Residuals)
    """
    residuals = y_test - y_pred
    
    plt.figure(figsize=(12, 5))
    
    # Residuals vs predicted values
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5, s=30)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Rainfall (mm)', fontsize=11)
    plt.ylabel('Residuals (mm)', fontsize=11)
    plt.title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Residuals distribution
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals (mm)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Distribution of Residuals', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"📊 Plot saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    # Đánh giá mô hình
    metrics = evaluate_model(
        csv_path="data/monthly_rainfall.csv",
        model_path="models/rainfall_model.pkl"
    )
    
    # Vẽ biểu đồ
    plot_predictions(metrics['y_test'], metrics['y_test_pred'], "models/predictions_plot.png")
    plot_residuals(metrics['y_test'], metrics['y_test_pred'], "models/residuals_plot.png")
    
    print("\n✅ Evaluation complete!")
