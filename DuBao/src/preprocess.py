import pandas as pd

def daily_to_monthly(input_file, output_file):
    # Đọc CSV - bỏ 2 dòng header lỗi
    df = pd.read_csv(input_file, skiprows=2)

    # In tên cột để kiểm tra
    print("Cột CSV đọc được:", df.columns.tolist())

    # Đổi tên cột đúng (Thờigian -> date, Pr_DaNang -> rainfall)
    df.columns = ["date", "rainfall"]

    # Convert date
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    
    # Convert rainfall thành số (loại bỏ giá trị không phải số)
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")

    # Kiểm tra dòng nào bị lỗi
    if df["date"].isna().sum() > 0:
        print("⚠ Có dòng không đọc được datetime, đang xoá các dòng lỗi...")
        df = df.dropna(subset=["date"])
    
    if df["rainfall"].isna().sum() > 0:
        print("⚠ Có dòng không đọc được rainfall, đang xoá các dòng lỗi...")
        df = df.dropna(subset=["rainfall"])

    # Lấy năm - tháng
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Gộp theo tháng - đảm bảo rainfall là số
    monthly = df.groupby(["year", "month"])["rainfall"].sum().reset_index()

    monthly.to_csv(output_file, index=False)
    print(f"✔ File tháng đã tạo: {output_file}")
    print("✓ Đã chuyển dữ liệu ngày → tháng thành công!")

def merge_weather_rainfall(rainfall_file, weather_file, output_file):
    """
    Merge rainfall data with weather data
    """
    # Load rainfall data
    rainfall_df = pd.read_csv(rainfall_file)
    rainfall_df['date'] = pd.to_datetime(rainfall_df[['year', 'month']].assign(day=1))

    # Load weather data
    weather_df = pd.read_csv(weather_file)
    weather_df['date'] = pd.to_datetime(weather_df['date'])

    # Aggregate weather data to monthly
    weather_df['year'] = weather_df['date'].dt.year
    weather_df['month'] = weather_df['date'].dt.month

    monthly_weather = weather_df.groupby(['year', 'month']).agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean'
    }).reset_index()

    # Merge data
    merged_df = pd.merge(rainfall_df, monthly_weather, on=['year', 'month'], how='inner')

    # Sort by date
    merged_df = merged_df.sort_values(['year', 'month']).reset_index(drop=True)

    merged_df.to_csv(output_file, index=False)
    print(f"✔ Merged data saved to {output_file}")
    print(f"Total records: {len(merged_df)}")
    print("✓ Đã merge dữ liệu mưa và thời tiết thành công!")