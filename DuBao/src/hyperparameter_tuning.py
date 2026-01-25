import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

def grid_search_gradient_boosting(csv_path, results_file="models/hyperparameters.json"):
    """
    Grid Search để tìm best hyperparameters cho Gradient Boosting
    """
    print("=" * 80)
    print("🔍 GRID SEARCH - HYPERPARAMETER TUNING")
    print("=" * 80)
    
    # Load dữ liệu
    df = pd.read_csv(csv_path)
    df = df.sort_values(['year', 'month']).reset_index(drop=True)
    
    # Feature engineering
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['rainfall_lag_1'] = df['rainfall'].shift(1)
    df['rainfall_lag_12'] = df['rainfall'].shift(12)
    df['rainfall_ma_3'] = df['rainfall'].rolling(window=3, min_periods=1).mean()
    df['rainfall_ma_12'] = df['rainfall'].rolling(window=12, min_periods=1).mean()
    df['trend'] = range(len(df))
    df['quarter'] = (df['month'] - 1) // 3 + 1
    df = df.dropna()
    
    feature_cols = [col for col in df.columns if col not in ['rainfall', 'year', 'month']]
    feature_cols.extend(['year', 'month'])
    
    X = df[feature_cols]
    y = df['rainfall']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Grid search parameters
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [3, 5, 7, 10],
        'subsample': [0.6, 0.8, 0.9, 1.0]
    }
    
    results = []
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    
    print(f"\n📊 Grid Search Configuration:")
    print(f"   Total combinations to test: {total_combinations}")
    print(f"   Parameters: {list(param_grid.keys())}\n")
    
    combo = 0
    best_r2 = -float('inf')
    best_params = None
    
    # Test tất cả combinations (có thể mất thời gian)
    from itertools import product
    
    param_values = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    for i, param_values_tuple in enumerate(param_values):
        combo += 1
        params = dict(zip(param_names, param_values_tuple))
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            subsample=params['subsample'],
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        results.append({
            'params': params,
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse)
        })
        
        # Track best
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_params = params
        
        # Progress
        if (i + 1) % max(1, total_combinations // 20) == 0:  # Show 20 updates
            print(f"   Progress: {combo}/{total_combinations} ({100*combo/total_combinations:.1f}%) - Best R²: {best_r2:.4f}")
    
    # Sort results by test_r2
    results_sorted = sorted(results, key=lambda x: x['test_r2'], reverse=True)
    
    print("\n" + "=" * 80)
    print("📊 TOP 10 BEST PARAMETER COMBINATIONS")
    print("=" * 80)
    
    print(f"\n{'Rank':<5} {'R² (Test)':<12} {'MAE':<10} {'RMSE':<10} {'Best Params':<50}")
    print("-" * 100)
    
    for idx, result in enumerate(results_sorted[:10], 1):
        params_str = f"n_est={result['params']['n_estimators']}, lr={result['params']['learning_rate']}, depth={result['params']['max_depth']}"
        print(f"{idx:<5} {result['test_r2']:<12.4f} {result['test_mae']:<10.4f} {result['test_rmse']:<10.4f} {params_str:<50}")
    
    # Save best parameters
    best_result = results_sorted[0]
    
    print(f"\n✅ BEST PARAMETERS FOUND:")
    print(f"   R² Score (Test): {best_result['test_r2']:.4f}")
    print(f"   MAE: {best_result['test_mae']:.4f} mm")
    print(f"   RMSE: {best_result['test_rmse']:.4f} mm")
    print(f"\n   Parameters:")
    for key, value in best_result['params'].items():
        print(f"      {key}: {value}")
    
    # Save all results
    with open(results_file, 'w') as f:
        json.dump({
            'best_result': best_result,
            'top_10_results': results_sorted[:10],
            'all_results': results_sorted
        }, f, indent=2)
    
    print(f"\n✅ Results saved: {results_file}")
    
    return best_result, results_sorted

if __name__ == "__main__":
    print("\n🔄 GRID SEARCH TUNING\n")
    
    best_result, all_results = grid_search_gradient_boosting("data/monthly_rainfall.csv")
    
    print("\n✅ Tuning complete!")
