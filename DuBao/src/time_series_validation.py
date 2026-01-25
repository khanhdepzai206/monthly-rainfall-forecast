import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

def test_stationarity(timeseries, name="Series"):
    """
    Kiểm tra tính stationarity bằng ADF test
    """
    print(f"\n{'='*60}")
    print(f"📊 ADF STATIONARITY TEST - {name}")
    print(f"{'='*60}")
    
    if not STATSMODELS_AVAILABLE:
        print("⚠️  statsmodels not available")
        return None
    
    result = adfuller(timeseries, autolag='AIC')
    
    print(f'\n📈 Test Results:')
    print(f'   ADF Test Statistic: {result[0]:.6f}')
    print(f'   P-value: {result[1]:.6f}')
    print(f'   Critical Values:')
    for key, value in result[4].items():
        print(f'      {key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print(f'\n✅ Result: Series is STATIONARY (p-value = {result[1]:.4f} < 0.05)')
        return True
    else:
        print(f'\n⚠️  Result: Series is NON-STATIONARY (p-value = {result[1]:.4f} >= 0.05)')
        return False

def analyze_seasonality(df):
    """
    Phân tích tính mùa vụ
    """
    print(f"\n{'='*60}")
    print(f"🌡️ SEASONALITY ANALYSIS")
    print(f"{'='*60}")
    
    # Group by month
    monthly_avg = df.groupby('month')['rainfall'].agg(['mean', 'std', 'min', 'max'])
    
    print(f'\n📊 Monthly Statistics:')
    print(monthly_avg.round(2).to_string())
    
    # Coefficient of variation by month
    monthly_avg['cv'] = monthly_avg['std'] / monthly_avg['mean']
    
    print(f'\n🔍 Seasonality Strength:')
    print(f'   Highest rainfall months: {monthly_avg["mean"].nlargest(3).index.tolist()}')
    print(f'   Lowest rainfall months: {monthly_avg["mean"].nsmallest(3).index.tolist()}')
    print(f'   Seasonal variation (CV): {monthly_avg["cv"].mean():.3f}')
    
    if monthly_avg["cv"].mean() > 0.5:
        print(f'   ✅ Strong seasonality detected!')
    else:
        print(f'   ⚠️  Weak seasonality')
    
    return monthly_avg

def data_quality_report(df):
    """
    Tạo báo cáo chất lượng dữ liệu
    """
    print(f"\n{'='*60}")
    print(f"📋 DATA QUALITY REPORT")
    print(f"{'='*60}")
    
    print(f'\n📊 Dataset Overview:')
    print(f'   Total records: {len(df)}')
    print(f'   Date range: {df["year"].min()}-{df["month"].min()} to {df["year"].max()}-{df["month"].max()}')
    print(f'   Missing values: {df["rainfall"].isna().sum()}')
    print(f'   Duplicate values: {df.duplicated().sum()}')
    
    print(f'\n💧 Rainfall Statistics:')
    print(f'   Mean: {df["rainfall"].mean():.2f} mm')
    print(f'   Median: {df["rainfall"].median():.2f} mm')
    print(f'   Std Dev: {df["rainfall"].std():.2f} mm')
    print(f'   Min: {df["rainfall"].min():.2f} mm')
    print(f'   Max: {df["rainfall"].max():.2f} mm')
    print(f'   IQR: {df["rainfall"].quantile(0.75) - df["rainfall"].quantile(0.25):.2f} mm')
    
    # Outlier detection (IQR method)
    Q1 = df["rainfall"].quantile(0.25)
    Q3 = df["rainfall"].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df["rainfall"] < Q1 - 1.5*IQR) | (df["rainfall"] > Q3 + 1.5*IQR)]
    
    print(f'\n🎯 Outlier Detection (IQR method):')
    print(f'   Outliers found: {len(outliers)} ({100*len(outliers)/len(df):.1f}%)')
    if len(outliers) > 0:
        print(f'   Outlier values: {outliers["rainfall"].values[:5].tolist()}...')
    
    # Data completeness
    expected_months = (df['year'].max() - df['year'].min() + 1) * 12
    actual_months = len(df)
    completeness = 100 * actual_months / expected_months
    
    print(f'\n✅ Data Completeness:')
    print(f'   Expected months: {expected_months}')
    print(f'   Actual months: {actual_months}')
    print(f'   Completeness: {completeness:.1f}%')
    
    if completeness > 95:
        print(f'   Status: ✅ EXCELLENT')
    elif completeness > 80:
        print(f'   Status: ✅ GOOD')
    else:
        print(f'   Status: ⚠️  NEEDS ATTENTION')
    
    return {
        'total_records': len(df),
        'missing_values': df["rainfall"].isna().sum(),
        'outliers': len(outliers),
        'completeness': completeness
    }

def plot_autocorrelation(timeseries, output_file="models/autocorrelation.png"):
    """
    Vẽ ACF/PACF plots
    """
    if not STATSMODELS_AVAILABLE:
        return
    
    print(f"\n📊 Generating autocorrelation plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # ACF
    plot_acf(timeseries, lags=40, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('ACF')
    
    # PACF
    plot_pacf(timeseries, lags=40, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('PACF')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✅ Plot saved: {output_file}")
    plt.close()

def validation_report(csv_path):
    """
    Main function - tạo toàn bộ validation report
    """
    print("\n" + "🔍 TIME SERIES VALIDATION REPORT ".center(60, "="))
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Tests
    is_stationary = test_stationarity(df['rainfall'].values, "Monthly Rainfall")
    seasonality = analyze_seasonality(df)
    quality = data_quality_report(df)
    
    # Plot autocorrelation
    plot_autocorrelation(df['rainfall'].values)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n✅ Data Quality:")
    print(f"   - Completeness: {quality['completeness']:.1f}%")
    print(f"   - Missing values: {quality['missing_values']}")
    print(f"   - Outliers: {quality['outliers']}")
    
    print(f"\n✅ Time Series Properties:")
    print(f"   - Stationary: {'✅ YES' if is_stationary else '⚠️  NO'}")
    print(f"   - Seasonal: {'✅ YES' if seasonality['mean'].max() / seasonality['mean'].min() > 1.5 else '⚠️  NO'}")
    
    print(f"\n✅ Recommendations:")
    if is_stationary:
        print(f"   - Use ARIMA(p,0,q) for modeling")
    else:
        print(f"   - Data needs differencing (d=1 or d=2)")
    
    if seasonality['mean'].max() / seasonality['mean'].min() > 1.5:
        print(f"   - Use SARIMA with seasonal component (s=12)")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    print("\n🔄 TIME SERIES VALIDATION\n")
    validation_report("data/monthly_rainfall.csv")
    print("✅ Validation complete!")
