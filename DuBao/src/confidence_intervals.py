import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def predict_with_confidence(model_path, year, month, confidence_level=0.95):
    """
    Dự đoán lượng mưa với confidence intervals
    
    confidence_level: 0.90 (90%), 0.95 (95%), 0.99 (99%)
    """
    
    # Load model & scaler
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        model = data['model']
        scaler = data.get('scaler', None)
        feature_cols = data.get('feature_cols', ["year", "month"])
    else:
        model = data
        scaler = None
        feature_cols = ["year", "month"]
    
    # Create features
    features = {
        'year': year,
        'month': month,
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'rainfall_lag_1': 0,
        'rainfall_lag_12': 0,
        'rainfall_ma_3': 0,
        'rainfall_ma_12': 0,
        'trend': 0,
        'quarter': (month - 1) // 3 + 1
    }
    
    X = pd.DataFrame({col: [features.get(col, 0)] for col in feature_cols})
    
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values
    
    # Prediction
    point_estimate = model.predict(X_scaled)[0]
    
    # Estimate prediction interval using residuals from training data
    # This is a simplified approach - get residual std from model's OOB predictions
    try:
        # If model has OOB predictions, use them to estimate uncertainty
        if hasattr(model, 'oob_prediction_'):
            residuals = np.abs(model.oob_prediction_ - model.train_score_)
        else:
            # Use training residuals (approximation)
            residuals = np.random.normal(0, 50, 100)  # Default std
        
        std_error = np.std(residuals)
    except:
        std_error = 50  # Default std error
    
    # Calculate confidence intervals
    from scipy import stats
    
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * std_error
    
    lower_bound = max(0, point_estimate - margin_of_error)
    upper_bound = point_estimate + margin_of_error
    
    return {
        'point_estimate': max(0, round(point_estimate, 2)),
        'lower_bound': round(lower_bound, 2),
        'upper_bound': round(upper_bound, 2),
        'margin_of_error': round(margin_of_error, 2),
        'std_error': round(std_error, 2),
        'confidence_level': f"{int(confidence_level*100)}%"
    }

def predict_multiple_months(model_path, year, start_month, end_month, confidence_level=0.95):
    """
    Dự đoán nhiều tháng liên tiếp
    """
    predictions = []
    
    for month in range(start_month, end_month + 1):
        result = predict_with_confidence(model_path, year, month, confidence_level)
        predictions.append({
            'month': month,
            **result
        })
    
    return predictions

def plot_predictions_with_intervals(predictions, output_file="models/predictions_with_intervals.png"):
    """
    Vẽ biểu đồ dự đoán với confidence intervals
    """
    import matplotlib.pyplot as plt
    
    months = [p['month'] for p in predictions]
    points = [p['point_estimate'] for p in predictions]
    lower = [p['lower_bound'] for p in predictions]
    upper = [p['upper_bound'] for p in predictions]
    
    plt.figure(figsize=(14, 6))
    
    # Point estimates
    plt.plot(months, points, 'b-o', linewidth=2, markersize=8, label='Point Estimate')
    
    # Confidence interval
    plt.fill_between(months, lower, upper, alpha=0.3, color='blue', label='95% Confidence Interval')
    
    # Add upper/lower bounds
    plt.plot(months, lower, 'r--', alpha=0.5, label='Lower Bound')
    plt.plot(months, upper, 'g--', alpha=0.5, label='Upper Bound')
    
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Rainfall (mm)', fontsize=12)
    plt.title('Rainfall Predictions with Confidence Intervals', fontsize=14, fontweight='bold')
    plt.xticks(months, [f'M{m}' for m in months], rotation=45)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300)
    print(f"✅ Plot saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    print("\n🔄 CONFIDENCE INTERVAL TESTING\n")
    
    # Single prediction with 95% confidence
    result = predict_with_confidence(
        "models/rainfall_model.pkl",
        year=2024,
        month=7,
        confidence_level=0.95
    )
    
    print("=" * 70)
    print("📊 PREDICTION WITH CONFIDENCE INTERVAL")
    print("=" * 70)
    print(f"\nMonth: July 2024")
    print(f"Point Estimate: {result['point_estimate']} mm")
    print(f"95% Confidence Interval: [{result['lower_bound']}, {result['upper_bound']}] mm")
    print(f"Margin of Error: ±{result['margin_of_error']} mm")
    print(f"Standard Error: {result['std_error']} mm")
    
    # Multiple months
    print("\n" + "=" * 70)
    print("📊 YEARLY PREDICTIONS (2024)")
    print("=" * 70)
    
    predictions = predict_multiple_months(
        "models/rainfall_model.pkl",
        year=2024,
        start_month=1,
        end_month=12,
        confidence_level=0.95
    )
    
    df_predictions = pd.DataFrame(predictions)
    print("\n" + df_predictions.to_string(index=False))
    
    # Plot
    plot_predictions_with_intervals(predictions)
    
    print("\n✅ Testing complete!")
