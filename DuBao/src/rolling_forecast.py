#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rolling forecast cho 4-5 ngày tiếp theo dựa trên:
- Dữ liệu lịch sử đến 01/04/2026
- Dữ liệu thời tiết dự báo từ Open-Meteo forecast API

Mô hình sklearn được train trên đủ 62 cột (xem train_daily_models.py + daily_features.csv).
Vector chỉ có các cột *_lag_* sẽ khiến predict() lỗi và bị bắt thành 0 mm — cần dựng đủ feature
theo feature_names_in_ và căn ngữ cảnh ngày (context = pred_date - 1) giống chuẩn bị dữ liệu.
"""
import requests
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date, timedelta


def fetch_forecast_weather(start_date, num_days=5, lat=16.0678, lon=108.2208):
    """
    Fetch dữ liệu thời tiết dự báo từ Open-Meteo.
    Dữ liệu có sẵn cho 7-10 ngày tiếp theo.
    """
    end_date = (pd.to_datetime(start_date) + timedelta(days=num_days)).strftime('%Y-%m-%d')

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
            "cloud_cover_mean",
            "surface_pressure_mean",
        ],
        "timezone": "Asia/Ho_Chi_Minh",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            daily = data["daily"]

            forecast_df = pd.DataFrame({
                "date": pd.to_datetime(daily["time"]),
                "temperature": (np.array(daily["temperature_2m_max"]) + np.array(daily["temperature_2m_min"])) / 2,
                "humidity": daily["relative_humidity_2m_mean"],
                "wind_speed": daily["wind_speed_10m_max"],
                "cloud_cover": daily["cloud_cover_mean"],
                "surface_pressure": daily["surface_pressure_mean"],
            })
            return forecast_df
        print(f"API error: {response.status_code}")
        return None
    except Exception as e:
        print(f"Error fetching forecast: {e}")
        return None


def _calendar_features(d: date) -> dict:
    ts = pd.Timestamp(d)
    month = int(ts.month)
    doy = int(ts.dayofyear)
    return {
        "year": int(ts.year),
        "month": month,
        "day": int(ts.day),
        "day_of_year": doy,
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "doy_sin": float(np.sin(2 * np.pi * doy / 365)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365)),
    }


def _rain_on(rain_by_date: dict, d: date) -> float:
    v = rain_by_date.get(d)
    return float(v) if v is not None else 0.0


def _build_weather_lookup(df_hist: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
    w = {}
    for _, row in df_hist.iterrows():
        d = row["date"]
        if not isinstance(d, date):
            d = pd.to_datetime(d).date()
        w[d] = {
            "temperature": float(row["temperature"]),
            "humidity": float(row["humidity"]),
            "wind_speed": float(row["wind_speed"]),
            "cloud_cover": float(row["cloud_cover"]),
            "surface_pressure": float(row["surface_pressure"]),
        }
    for _, row in forecast_df.iterrows():
        d = row["date"]
        d = d.date() if hasattr(d, "date") else pd.to_datetime(d).date()
        w[d] = {
            "temperature": float(row["temperature"]),
            "humidity": float(row["humidity"]),
            "wind_speed": float(row["wind_speed"]),
            "cloud_cover": float(row["cloud_cover"]),
            "surface_pressure": float(row["surface_pressure"]),
        }
    return w


def _weather_at(weather_by_date: dict, d: date, fallback: dict) -> dict:
    if d in weather_by_date:
        return weather_by_date[d]
    return fallback


def rolling_forecast(start_date, num_days=5, models_dir='DuBao/models', data_dir='DuBao/data'):
    """
    Dự báo rolling cho num_days ngày tiếp theo.

    Mỗi pred_date: dự đoán mưa *pred_date* dùng feature row tương ứng ngày ngữ cảnh ctx = pred_date - 1
    (giống daily_features: target trên ngày D là mưa ngày D+1).
    """
    print(f"Rolling forecast tu {start_date} cho {num_days} ngay")

    historical_path = os.path.join(data_dir, 'daily_features.csv')
    df_hist = pd.read_csv(historical_path)
    df_hist['date'] = pd.to_datetime(df_hist['date']).dt.date

    models_data = {
        'RF': joblib.load(os.path.join(models_dir, 'rf_daily_model.pkl')),
        'XGB': joblib.load(os.path.join(models_dir, 'xgb_daily_model.pkl')),
        'LR': joblib.load(os.path.join(models_dir, 'lr_daily_model.pkl')),
    }
    ref_model = models_data['RF']
    feature_names = list(ref_model.feature_names_in_)

    start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    forecast_df = fetch_forecast_weather(start_date_str, num_days)
    if forecast_df is None:
        print("Khong lay duoc du lieu forecast thoi tiet")
        return None

    last_hist_date = max(df_hist['date'])
    last_hist_row = df_hist[df_hist['date'] == last_hist_date].iloc[0]
    fallback_w = {
        "temperature": float(last_hist_row["temperature"]),
        "humidity": float(last_hist_row["humidity"]),
        "wind_speed": float(last_hist_row["wind_speed"]),
        "cloud_cover": float(last_hist_row["cloud_cover"]),
        "surface_pressure": float(last_hist_row["surface_pressure"]),
    }

    weather_by_date = _build_weather_lookup(df_hist, forecast_df)

    rain_by_date = {}
    for _, row in df_hist.iterrows():
        d = row["date"]
        if not isinstance(d, date):
            d = pd.to_datetime(d).date()
        rain_by_date[d] = float(row["rainfall"])

    predictions = []
    forecast_slice = forecast_df.head(num_days)

    for _, forecast_row in forecast_slice.iterrows():
        pred_date = pd.to_datetime(forecast_row['date']).date()
        ctx = pred_date - timedelta(days=1)

        w_ctx = _weather_at(weather_by_date, ctx, fallback_w)
        cal = _calendar_features(ctx)
        rain_ctx = _rain_on(rain_by_date, ctx)

        feat = {
            "temperature": w_ctx["temperature"],
            "humidity": w_ctx["humidity"],
            "wind_speed": w_ctx["wind_speed"],
            "cloud_cover": w_ctx["cloud_cover"],
            "surface_pressure": w_ctx["surface_pressure"],
            **cal,
            "is_wet": int(rain_ctx > 0),
        }

        for k in range(1, 8):
            dlag = ctx - timedelta(days=k)
            wlag = _weather_at(weather_by_date, dlag, fallback_w)
            feat[f"temperature_lag_{k}"] = wlag["temperature"]
            feat[f"humidity_lag_{k}"] = wlag["humidity"]
            feat[f"wind_speed_lag_{k}"] = wlag["wind_speed"]
            feat[f"cloud_cover_lag_{k}"] = wlag["cloud_cover"]
            feat[f"surface_pressure_lag_{k}"] = wlag["surface_pressure"]
            feat[f"rainfall_lag_{k}"] = _rain_on(rain_by_date, dlag)

        r_m1 = _rain_on(rain_by_date, ctx - timedelta(days=1))
        r_m2 = _rain_on(rain_by_date, ctx - timedelta(days=2))
        r_m3 = _rain_on(rain_by_date, ctx - timedelta(days=3))
        feat["rainfall_ma_3"] = float(np.mean([r_m1, r_m2, r_m3]))
        r7 = [_rain_on(rain_by_date, ctx - timedelta(days=i)) for i in range(1, 8)]
        feat["rainfall_ma_7"] = float(np.mean(r7))

        def _wet(d0: date) -> int:
            return int(_rain_on(rain_by_date, d0) > 0)

        feat["wet_spell_3"] = float(
            _wet(ctx - timedelta(days=1)) + _wet(ctx - timedelta(days=2)) + _wet(ctx - timedelta(days=3))
        )
        feat["wet_spell_7"] = float(sum(_wet(ctx - timedelta(days=i)) for i in range(1, 8)))

        t_m3 = [
            _weather_at(weather_by_date, ctx - timedelta(days=i), fallback_w)["temperature"]
            for i in (1, 2, 3)
        ]
        h_m3 = [
            _weather_at(weather_by_date, ctx - timedelta(days=i), fallback_w)["humidity"]
            for i in (1, 2, 3)
        ]
        feat["temperature_ma_3"] = float(np.mean(t_m3))
        feat["humidity_ma_3"] = float(np.mean(h_m3))

        row = {n: feat[n] for n in feature_names}
        feature_vector = pd.DataFrame([row], columns=feature_names)

        model_preds = {}
        for model_key, model in models_data.items():
            try:
                pred = float(model.predict(feature_vector)[0])
                model_preds[model_key] = max(0.0, pred)
            except Exception as e:
                print(f"  Error predicting with {model_key}: {e}")
                model_preds[model_key] = 0.0

        chain = model_preds['RF']
        rain_by_date[pred_date] = chain

        disp_temp = float(forecast_row["temperature"])
        disp_hum = float(forecast_row["humidity"])
        disp_wind = float(forecast_row["wind_speed"])

        predictions.append({
            'date': pred_date,
            'date_str': pred_date.strftime('%d/%m/%Y'),
            'RF': round(model_preds['RF'], 2),
            'XGB': round(model_preds['XGB'], 2),
            'LR': round(model_preds['LR'], 2),
            'forecast_temp': round(disp_temp, 1),
            'forecast_humidity': round(disp_hum, 1),
            'forecast_wind': round(disp_wind, 1),
        })

        print(f"  OK {pred_date}: RF={model_preds['RF']:.2f}mm, XGB={model_preds['XGB']:.2f}mm, LR={model_preds['LR']:.2f}mm")

    return predictions


if __name__ == "__main__":
    predictions = rolling_forecast('2026-04-02', num_days=5)

    if predictions:
        print("\n" + "=" * 60)
        print("ROLLING FORECAST RESULTS")
        print("=" * 60)
        for pred in predictions:
            print(f"{pred['date_str']}: RF={pred['RF']}mm, XGB={pred['XGB']}mm, LR={pred['LR']}mm")
