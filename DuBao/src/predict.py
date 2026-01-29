import pickle
import os
import pandas as pd
import numpy as np

# Năm gốc dữ liệu train (trend = số tháng từ đây)
BASE_YEAR = 1979


def _get_rainfall_lags_from_history(weather_data_path, year, month):
    """
    Lấy rainfall lag/ma từ dữ liệu lịch sử nếu có; không thì dùng trung bình theo tháng.
    Giúp dự đoán khác nhau theo năm khi có (year-1, month) hoặc (year, month-1).
    """
    out = {'rainfall_lag_1': None, 'rainfall_lag_12': None, 'rainfall_ma_3': None, 'rainfall_ma_12': None}
    if not weather_data_path:
        return out
    try:
        df = pd.read_csv(weather_data_path)
        if 'rainfall' not in df.columns or 'year' not in df.columns or 'month' not in df.columns:
            return out
        monthly_rain = df.groupby('month')['rainfall'].mean()
        default_ma12 = float(monthly_rain.loc[month]) if month in monthly_rain.index else float(monthly_rain.mean())
        out['rainfall_ma_12'] = default_ma12
        out['rainfall_lag_12'] = default_ma12
        m_prev = month - 1 if month > 1 else 12
        out['rainfall_lag_1'] = float(monthly_rain.loc[m_prev]) if m_prev in monthly_rain.index else default_ma12
        m2 = month - 2 if month >= 3 else month + 10
        m3 = month - 3 if month >= 4 else month + 9
        out['rainfall_ma_3'] = (out['rainfall_lag_1'] + float(monthly_rain.loc[m2]) + float(monthly_rain.loc[m3])) / 3.0 if m2 in monthly_rain.index and m3 in monthly_rain.index else default_ma12
        row_12 = df[(df['year'] == year - 1) & (df['month'] == month)]
        if len(row_12) > 0:
            out['rainfall_lag_12'] = float(row_12['rainfall'].iloc[0])
        if month > 1:
            row_1 = df[(df['year'] == year) & (df['month'] == month - 1)]
        else:
            row_1 = df[(df['year'] == year - 1) & (df['month'] == 12)]
        if len(row_1) > 0:
            out['rainfall_lag_1'] = float(row_1['rainfall'].iloc[0])
        recent = []
        for dm in [-1, -2, -3]:
            m, y = month + dm, year
            if m < 1:
                m, y = m + 12, y - 1
            elif m > 12:
                m, y = m - 12, y + 1
            r = df[(df['year'] == y) & (df['month'] == m)]
            if len(r) > 0:
                recent.append(float(r['rainfall'].iloc[0]))
        if len(recent) >= 2:
            out['rainfall_ma_3'] = float(np.mean(recent))
    except Exception:
        pass
    return out


def _year_adjustment(weather_data_path, year, month, base_pred):
    """
    Hiệu chỉnh dự đoán theo năm từ xu hướng lịch sử.
    Áp dụng cho mọi tháng (1-12): cùng tháng khác năm -> giá trị khác nhau.
    """
    fallback = lambda: float(base_pred) * (1 + 0.004 * (year - 2020))
    try:
        if not weather_data_path:
            return fallback()
        df = pd.read_csv(weather_data_path)
        if 'year' not in df.columns or 'rainfall' not in df.columns or 'month' not in df.columns:
            return fallback()
        # Lọc đúng tháng (1-12) để tính xu hướng theo năm
        same_month = df[df['month'] == int(month)][['year', 'rainfall']]
        if len(same_month) < 5:
            return fallback()
        by_year = same_month.groupby('year')['rainfall'].mean()
        if len(by_year) < 2:
            return fallback()
        x = np.array(by_year.index, dtype=float)
        y = np.array(by_year.values, dtype=float)
        x_mean = x.mean()
        denom = np.sum((x - x_mean) ** 2) + 1e-8
        slope = np.sum((x - x_mean) * (y - np.mean(y))) / denom
        ref_year = float(x_mean)
        return float(base_pred) + slope * (year - ref_year)
    except Exception:
        return fallback()


def predict_rainfall(model_path, year, month, weather_data_path=None):
    """
    Dự đoán lượng mưa sử dụng mô hình đã train.

    - Mọi loại mô hình: Gradient Boosting, Random Forest (dict hoặc legacy), SARIMA đều
      cho kết quả khác nhau theo từng năm (cùng tháng khác năm -> khác mm).
    - Mọi tháng 1-12: trend/lags và _year_adjustment dùng đúng tháng tương ứng.
    """
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and 'model' in data:
        model_obj = data['model']
        if hasattr(model_obj, 'get_forecast'):  # SARIMA/SARIMAX: theo tháng là giống nhau -> thêm hiệu chỉnh theo năm
            try:
                if weather_data_path:
                    weather_df = pd.read_csv(weather_data_path)
                    monthly_avg = weather_df[weather_df['month'] == month]['rainfall'].mean()
                    base = monthly_avg if not pd.isna(monthly_avg) else 100
                else:
                    base = 100
                prediction = _year_adjustment(weather_data_path, year, month, float(base))
            except Exception as e:
                print(f"SARIMA prediction error: {e}")
                prediction = _year_adjustment(weather_data_path, year, month, 100.0)
        elif hasattr(model_obj, 'predict'):  # sklearn models
            scaler = data.get('scaler', None)
            feature_cols = data.get('feature_cols', ["year", "month"])
            base_year = int(data.get('base_year', BASE_YEAR))

            # Trend = số tháng từ năm gốc (để 7/2021 ≠ 7/2022)
            trend = (year - base_year) * 12 + (month - 1)
            trend = max(0, trend)

            # Lags từ lịch sử hoặc trung bình theo tháng (không còn toàn 0)
            lags = _get_rainfall_lags_from_history(weather_data_path, year, month)
            rainfall_lag_1 = lags['rainfall_lag_1'] if lags['rainfall_lag_1'] is not None else 0
            rainfall_lag_12 = lags['rainfall_lag_12'] if lags['rainfall_lag_12'] is not None else 0
            rainfall_ma_3 = lags['rainfall_ma_3'] if lags['rainfall_ma_3'] is not None else 0
            rainfall_ma_12 = lags['rainfall_ma_12'] if lags['rainfall_ma_12'] is not None else 0

            weather_features = {}
            if weather_data_path and 'temperature' in feature_cols:
                weather_df = pd.read_csv(weather_data_path)
                agg_dict = {'temperature': 'mean', 'humidity': 'mean', 'wind_speed': 'mean'}
                if 'cloud_cover' in weather_df.columns:
                    agg_dict['cloud_cover'] = 'mean'
                if 'surface_pressure' in weather_df.columns:
                    agg_dict['surface_pressure'] = 'mean'
                monthly_avg = weather_df.groupby('month').agg(agg_dict).loc[month]
                weather_features = {
                    'temperature': monthly_avg['temperature'],
                    'temp_lag_1': monthly_avg['temperature'],
                    'temp_lag_12': monthly_avg['temperature'],
                    'temp_ma_3': monthly_avg['temperature'],
                    'humidity': monthly_avg['humidity'],
                    'humidity_lag_1': monthly_avg['humidity'],
                    'humidity_lag_12': monthly_avg['humidity'],
                    'humidity_ma_3': monthly_avg['humidity'],
                    'wind_speed': monthly_avg['wind_speed'],
                    'wind_lag_1': monthly_avg['wind_speed'],
                    'wind_lag_12': monthly_avg['wind_speed'],
                    'wind_ma_3': monthly_avg['wind_speed']
                }
                if 'cloud_cover' in monthly_avg:
                    c = monthly_avg['cloud_cover']
                    weather_features['cloud_cover'] = weather_features['cloud_cover_lag_1'] = weather_features['cloud_cover_ma_3'] = c
                if 'surface_pressure' in monthly_avg:
                    p = monthly_avg['surface_pressure']
                    weather_features['surface_pressure'] = weather_features['surface_pressure_lag_1'] = weather_features['surface_pressure_ma_3'] = p

            features = {
                'year': year,
                'month': month,
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                'rainfall_lag_1': rainfall_lag_1,
                'rainfall_lag_12': rainfall_lag_12,
                'rainfall_ma_3': rainfall_ma_3,
                'rainfall_ma_12': rainfall_ma_12,
                'trend': trend,
                'quarter': (month - 1) // 3 + 1,
                **weather_features
            }

            X = pd.DataFrame({col: [features.get(col, 0)] for col in feature_cols})
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X.values
            prediction = model_obj.predict(X_scaled)[0]
        else:
            # Dict nhưng model chỉ nhận year, month -> vẫn hiệu chỉnh theo năm để khác năm ra khác số
            X = pd.DataFrame({"year": [year], "month": [month]})
            base = model_obj.predict(X)[0]
            prediction = _year_adjustment(weather_data_path, year, month, float(base))
    else:
        # Pickle cũ: model là object trực tiếp (chỉ có year, month) -> thêm hiệu chỉnh theo năm
        if hasattr(data, 'get_forecast'):  # SARIMA
            forecast = data.get_forecast(steps=1)
            base = forecast.predicted_mean.values[0]
            prediction = _year_adjustment(weather_data_path, year, month, float(base))
        else:
            X_legacy = pd.DataFrame({"year": [year], "month": [month]})
            base = data.predict(X_legacy)[0]
            prediction = _year_adjustment(weather_data_path, year, month, float(base))

    return max(0, prediction)  # Không âm


def _daily_features_from_history(daily_data_path, date_obj, feature_cols, base_year=1979):
    """Tạo dict feature cho một ngày từ daily_combined (lags từ lịch sử hoặc TB theo ngày trong năm)."""
    from datetime import timedelta
    row = {
        "year": date_obj.year,
        "month": date_obj.month,
        "day": date_obj.day,
        "day_of_year": date_obj.timetuple().tm_yday,
        "doy_sin": np.sin(2 * np.pi * date_obj.timetuple().tm_yday / 365),
        "doy_cos": np.cos(2 * np.pi * date_obj.timetuple().tm_yday / 365),
        "trend": (date_obj.year - base_year) * 365 + date_obj.timetuple().tm_yday,
    }
    row["rainfall_lag_1"] = row["rainfall_lag_7"] = row["rainfall_lag_30"] = row["rainfall_ma_7"] = 0.0
    row["temp_lag_1"] = row["temp_ma_7"] = row["humidity_lag_1"] = row["wind_lag_1"] = 0.0
    if not daily_data_path:
        for c in feature_cols:
            if c not in row:
                row[c] = 0
        return row
    try:
        df = pd.read_csv(daily_data_path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date")
        day_mean = df.groupby(df["date"].dt.dayofyear)["rainfall"].mean()
        doy = date_obj.timetuple().tm_yday
        default_rain = float(day_mean.loc[doy]) if doy in day_mean.index else float(df["rainfall"].mean())
        row["rainfall_lag_1"] = row["rainfall_lag_7"] = row["rainfall_lag_30"] = row["rainfall_ma_7"] = default_rain
        for d in [1, 7, 30]:
            prev = date_obj - timedelta(days=d)
            p = df[df["date"] == pd.Timestamp(prev)]
            if len(p) > 0:
                if d == 1:
                    row["rainfall_lag_1"] = float(p["rainfall"].iloc[0])
                elif d == 7:
                    row["rainfall_lag_7"] = float(p["rainfall"].iloc[0])
                else:
                    row["rainfall_lag_30"] = float(p["rainfall"].iloc[0])
        last7 = df[df["date"] < pd.Timestamp(date_obj)].tail(7)
        if len(last7) >= 2:
            row["rainfall_ma_7"] = float(last7["rainfall"].mean())
        if "temperature" in df.columns:
            t_avg = df.groupby(df["date"].dt.dayofyear)["temperature"].mean()
            row["temp_lag_1"] = row["temp_ma_7"] = float(t_avg.loc[doy]) if doy in t_avg.index else float(df["temperature"].mean())
            p1 = df[df["date"] == pd.Timestamp(date_obj - timedelta(days=1))]
            if len(p1) > 0:
                row["temp_lag_1"] = float(p1["temperature"].iloc[0])
        if "humidity" in df.columns:
            h_avg = df.groupby(df["date"].dt.dayofyear)["humidity"].mean()
            row["humidity_lag_1"] = float(h_avg.loc[doy]) if doy in h_avg.index else float(df["humidity"].mean())
        if "wind_speed" in df.columns:
            w_avg = df.groupby(df["date"].dt.dayofyear)["wind_speed"].mean()
            row["wind_lag_1"] = float(w_avg.loc[doy]) if doy in w_avg.index else float(df["wind_speed"].mean())
    except Exception:
        pass
    for c in feature_cols:
        if c not in row:
            row[c] = 0
    return row


def predict_rainfall_daily(model_path, year, month, day, daily_data_path=None):
    """
    Dự đoán lượng mưa theo NGÀY (mm/ngày).
    model_path: path tới daily_rainfall_model.pkl
    daily_data_path: path tới daily_combined.csv (để lấy lags/weather).
    """
    from datetime import date
    date_obj = date(int(year), int(month), int(day))
    if not os.path.exists(model_path):
        return 0.0
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "model" not in data:
        return 0.0
    model = data["model"]
    scaler = data.get("scaler")
    feature_cols = data.get("feature_cols", [])
    base_year = int(data.get("base_year", BASE_YEAR))
    feats = _daily_features_from_history(daily_data_path, date_obj, feature_cols, base_year)
    X = pd.DataFrame({c: [feats.get(c, 0)] for c in feature_cols})
    if scaler is not None:
        X = scaler.transform(X)
    else:
        X = X.values
    pred = model.predict(X)[0]
    return max(0.0, float(pred))

