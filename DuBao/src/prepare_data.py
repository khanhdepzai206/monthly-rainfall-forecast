from preprocess import daily_to_monthly, merge_weather_rainfall

def main():
    # Convert daily rainfall to monthly
    daily_to_monthly("../data/raw_daily.csv", "../data/monthly_rainfall.csv")

    # Merge with weather data
    merge_weather_rainfall("../data/monthly_rainfall.csv", "../data/weather_daily.csv", "../data/monthly_combined.csv")

if __name__ == "__main__":
    main()