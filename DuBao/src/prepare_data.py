from preprocess import daily_to_monthly, merge_weather_rainfall, create_daily_combined

def main():
    # Convert daily rainfall to monthly
    daily_to_monthly("../data/raw_daily.csv", "../data/monthly_rainfall.csv")
    # Merge monthly with weather
    merge_weather_rainfall("../data/monthly_rainfall.csv", "../data/weather_daily.csv", "../data/monthly_combined.csv")
    # Dữ liệu theo ngày (mưa + thời tiết) cho mô hình dự đoán theo ngày
    create_daily_combined("../data/raw_daily.csv", "../data/weather_daily.csv", "../data/daily_combined.csv")


def prepare_data_daily_only():
    """Chỉ tạo daily_combined (khi đã có raw_daily và weather_daily)."""
    create_daily_combined("../data/raw_daily.csv", "../data/weather_daily.csv", "../data/daily_combined.csv")


if __name__ == "__main__":
    main()