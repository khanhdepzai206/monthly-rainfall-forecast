import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

def analyze_daily_data(csv_path, output_dir="models"):
    """
    Phân tích dữ liệu hàng ngày
    """
    # Load dữ liệu
    df = pd.read_csv(csv_path, skiprows=2)
    df.columns = ["date", "rainfall"]
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["date"])
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
    
    print("=" * 60)
    print("📊 DAILY DATA ANALYSIS")
    print("=" * 60)
    
    print(f"\n📈 Dataset Overview:")
    print(f"   Total records: {len(df):,}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Duration: {(df['date'].max() - df['date'].min()).days} days")
    
    print(f"\n💧 Rainfall Statistics (Daily):")
    print(f"   Mean: {df['rainfall'].mean():.2f} mm")
    print(f"   Median: {df['rainfall'].median():.2f} mm")
    print(f"   Std Dev: {df['rainfall'].std():.2f} mm")
    print(f"   Min: {df['rainfall'].min():.2f} mm")
    print(f"   Max: {df['rainfall'].max():.2f} mm")
    print(f"   25th percentile: {df['rainfall'].quantile(0.25):.2f} mm")
    print(f"   75th percentile: {df['rainfall'].quantile(0.75):.2f} mm")
    
    # Days with rain
    rainy_days = (df['rainfall'] > 0).sum()
    print(f"\n☔ Rainy Days:")
    print(f"   Days with rain: {rainy_days} ({rainy_days/len(df)*100:.1f}%)")
    print(f"   Days without rain: {len(df) - rainy_days} ({(len(df)-rainy_days)/len(df)*100:.1f}%)")
    
    # Extract year and month for monthly analysis
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%B')
    
    # Monthly statistics
    monthly_stats = df.groupby('month').agg({
        'rainfall': ['sum', 'mean', 'std', 'max', 'min']
    }).round(2)
    
    print(f"\n📅 Monthly Average Rainfall:")
    for month in range(1, 13):
        month_data = df[df['month'] == month]['rainfall']
        if len(month_data) > 0:
            print(f"   Month {month:2d}: {month_data.sum():8.2f} mm (avg: {month_data.mean():6.2f} mm)")
    
    print("\n" + "=" * 60)
    
    return df

def analyze_monthly_data(csv_path):
    """
    Phân tích dữ liệu hàng tháng
    """
    df = pd.read_csv(csv_path)
    
    print("=" * 60)
    print("📊 MONTHLY DATA ANALYSIS")
    print("=" * 60)
    
    print(f"\n📈 Dataset Overview:")
    print(f"   Total months: {len(df)}")
    print(f"   Year range: {df['year'].min()} to {df['year'].max()}")
    
    print(f"\n💧 Monthly Rainfall Statistics:")
    print(f"   Mean: {df['rainfall'].mean():.2f} mm")
    print(f"   Median: {df['rainfall'].median():.2f} mm")
    print(f"   Std Dev: {df['rainfall'].std():.2f} mm")
    print(f"   Min: {df['rainfall'].min():.2f} mm")
    print(f"   Max: {df['rainfall'].max():.2f} mm")
    
    # Wettest and driest months
    wettest = df.loc[df['rainfall'].idxmax()]
    driest = df.loc[df['rainfall'].idxmin()]
    
    print(f"\n☔ Extreme Months:")
    print(f"   Wettest: {int(wettest['month'])}/{int(wettest['year'])} ({wettest['rainfall']:.2f} mm)")
    print(f"   Driest: {int(driest['month'])}/{int(driest['year'])} ({driest['rainfall']:.2f} mm)")
    
    # Seasonal patterns
    print(f"\n🌡️ Seasonal Patterns (Average Monthly Rainfall):")
    seasonal = df.groupby('month')['rainfall'].agg(['mean', 'std'])
    for month in range(1, 13):
        if month in seasonal.index:
            print(f"   Month {month:2d}: {seasonal.loc[month, 'mean']:7.2f} ± {seasonal.loc[month, 'std']:5.2f} mm")
    
    print("\n" + "=" * 60)
    
    return df

def plot_daily_rainfall_distribution(csv_path, output_file="models/daily_distribution.png"):
    """
    Vẽ biểu đồ phân phối lượng mưa hàng ngày
    """
    df = pd.read_csv(csv_path, skiprows=2)
    df.columns = ["date", "rainfall"]
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
    df = df.dropna(subset=["rainfall"])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    axes[0, 0].hist(df['rainfall'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Daily Rainfall (mm)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Daily Rainfall', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Box plot
    axes[0, 1].boxplot(df['rainfall'], vert=True)
    axes[0, 1].set_ylabel('Rainfall (mm)')
    axes[0, 1].set_title('Box Plot of Daily Rainfall', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Log scale histogram
    rainfall_nonzero = df[df['rainfall'] > 0]['rainfall']
    axes[1, 0].hist(rainfall_nonzero, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[1, 0].set_xlabel('Daily Rainfall (mm)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Rainy Days (rainfall > 0mm)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Cumulative distribution
    sorted_rainfall = np.sort(df['rainfall'])
    cumsum = np.cumsum(sorted_rainfall)
    axes[1, 1].plot(sorted_rainfall, cumsum / cumsum[-1], linewidth=2, color='green')
    axes[1, 1].set_xlabel('Daily Rainfall (mm)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution of Daily Rainfall', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"📊 Plot saved: {output_file}")
    plt.close()

def plot_monthly_time_series(csv_path, output_file="models/monthly_timeseries.png"):
    """
    Vẽ biểu đồ chuỗi thời gian lượng mưa hàng tháng
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series
    axes[0].plot(df['date'], df['rainfall'], linewidth=1.5, color='blue', alpha=0.7)
    axes[0].fill_between(df['date'], df['rainfall'], alpha=0.3, color='blue')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Monthly Rainfall (mm)')
    axes[0].set_title('Time Series of Monthly Rainfall', fontweight='bold', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Seasonal pattern (by month)
    monthly_avg = df.groupby('month')['rainfall'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    axes[1].bar(range(1, 13), monthly_avg.values, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(months)
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Average Rainfall (mm)')
    axes[1].set_title('Seasonal Pattern - Average Monthly Rainfall', fontweight='bold', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"📊 Plot saved: {output_file}")
    plt.close()

def plot_monthly_heatmap(csv_path, output_file="models/monthly_heatmap.png"):
    """
    Vẽ biểu đồ heatmap: rainfall by year và month
    """
    df = pd.read_csv(csv_path)
    
    # Pivot table: years x months
    pivot = df.pivot_table(values='rainfall', index='year', columns='month', aggfunc='mean')
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot, cmap='YlGnBu', cbar_kws={'label': 'Rainfall (mm)'}, 
                linewidths=0.5, fmt='.1f')
    plt.title('Rainfall Heatmap: Year vs Month', fontweight='bold', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"📊 Plot saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    print("\n🔍 EXPLORATORY DATA ANALYSIS\n")
    
    # Analyze daily data
    daily_df = analyze_daily_data("data/raw_daily.csv")
    
    print("\n")
    
    # Analyze monthly data
    monthly_df = analyze_monthly_data("data/monthly_rainfall.csv")
    
    print("\n📊 Creating visualizations...\n")
    
    # Create plots
    plot_daily_rainfall_distribution("data/raw_daily.csv", "models/daily_distribution.png")
    plot_monthly_time_series("data/monthly_rainfall.csv", "models/monthly_timeseries.png")
    plot_monthly_heatmap("data/monthly_rainfall.csv", "models/monthly_heatmap.png")
    
    print("\n✅ EDA complete!")
