import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_weather_data(start_date, end_date, lat=16.0678, lon=108.2208):
    """
    Lấy dữ liệu thời tiết lịch sử từ Open-Meteo API (Đà Nẵng).
    Gồm: nhiệt độ, độ ẩm, gió, mây che phủ, áp suất mặt đất để tăng độ chính xác mô hình.
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    daily_vars = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_mean",
        "cloud_cover_mean",
        "surface_pressure_mean",
    ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": daily_vars,
        "timezone": "Asia/Ho_Chi_Minh",
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        d = data["daily"]
        df = pd.DataFrame({
            "date": d["time"],
            "temperature": d["temperature_2m_mean"],
            "humidity": d["relative_humidity_2m_mean"],
            "wind_speed": d["wind_speed_10m_mean"],
            "cloud_cover": d["cloud_cover_mean"],
            "surface_pressure": d["surface_pressure_mean"],
        })
        df["date"] = pd.to_datetime(df["date"])
        return df
    else:
        print(f"Lỗi API: {response.status_code}")
        return None

def main():
    start_date = "1979-01-01"
    end_date = "2022-12-31"

    print("Đang gọi Open-Meteo API lấy dữ liệu thời tiết (nhiệt độ, độ ẩm, gió, mây, áp suất)...")
    weather_df = fetch_weather_data(start_date, end_date)

    if weather_df is not None:
        output_file = "../data/weather_daily.csv"
        weather_df.to_csv(output_file, index=False)
        print(f"Đã lưu: {output_file}, số bản ghi: {len(weather_df)}")
        print("Cột:", list(weather_df.columns))
    else:
        print("Không lấy được dữ liệu từ API.")

if __name__ == "__main__":
    main()