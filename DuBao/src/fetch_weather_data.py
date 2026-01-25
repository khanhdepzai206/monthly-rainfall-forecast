import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_weather_data(start_date, end_date, lat=16.0678, lon=108.2208):
    """
    Fetch historical weather data from Open-Meteo API
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"

    # Parameters for Đà Nẵng
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_mean"],
        "timezone": "Asia/Ho_Chi_Minh"
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame({
            'date': data['daily']['time'],
            'temperature': data['daily']['temperature_2m_mean'],
            'humidity': data['daily']['relative_humidity_2m_mean'],
            'wind_speed': data['daily']['wind_speed_10m_mean']
        })
        df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

def main():
    # Fetch data from 1979 to 2022
    start_date = "1979-01-01"
    end_date = "2022-12-31"

    print("Fetching weather data from Open-Meteo API...")
    weather_df = fetch_weather_data(start_date, end_date)

    if weather_df is not None:
        # Save to CSV
        output_file = "../data/weather_daily.csv"
        weather_df.to_csv(output_file, index=False)
        print(f"Weather data saved to {output_file}")
        print(f"Total records: {len(weather_df)}")
    else:
        print("Failed to fetch weather data")

if __name__ == "__main__":
    main()