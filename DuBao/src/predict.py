import pickle
import os
import pandas as pd
import numpy as np

# Năm gốc dữ liệu train (trend = số tháng từ đây)
BASE_YEAR = 1979


def _map_calendar_to_history_date(ts, df):
    """
    Ánh xạ một ngày (kể cả năm tương lai) sang một ngày nằm trong khoảng dữ liệu CSV
    bằng cách lặp theo các năm có sẵn, để cùng tháng/ngày nhưng khác năm dự đoán
    → khác năm trong lịch sử → lag / MA khác nhau.
    """
    ts = pd.Timestamp(ts).normalize()
    dmin = df["date"].min()
    dmax = df["date"].max()
    if dmin <= ts <= dmax:
        return ts
    years = sorted(df["date"].dt.year.unique().tolist())
    if not years:
        return ts
    n = len(years)
    yi = (int(ts.year) - years[0]) % n
    mapped_y = years[yi]
    try:
        return pd.Timestamp(year=mapped_y, month=ts.month, day=ts.day)
    except ValueError:
        return pd.Timestamp(year=mapped_y, month=ts.month, day=28)


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


def _build_daily_features_for_date(date_obj, daily_data_path, feature_cols, base_year=1979):
    """Tạo đầy đủ features cho 1 ngày (tương thích với train_daily_two_stage.build_features)."""
    from datetime import timedelta
    doy = date_obj.timetuple().tm_yday
    row = {
        "year": date_obj.year,
        "month": date_obj.month,
        "day": date_obj.day,
        "day_of_year": doy,
        "doy_sin": np.sin(2 * np.pi * doy / 365),
        "doy_cos": np.cos(2 * np.pi * doy / 365),
        "month_sin": np.sin(2 * np.pi * date_obj.month / 12),
        "month_cos": np.cos(2 * np.pi * date_obj.month / 12),
        "trend": (date_obj.year - base_year) * 365 + doy,
    }
    for lag in [1, 2, 7, 14, 30]:
        row[f"rainfall_lag_{lag}"] = 0.0
    for w in [3, 7, 14]:
        row[f"rainfall_ma_{w}"] = 0.0
    for col in ["temperature", "humidity", "wind_speed"]:
        row[f"{col}_lag_1"] = row[f"{col}_ma_7"] = 0.0
    row["cloud_cover_lag_1"] = row["surface_pressure_lag_1"] = 0.0

    if daily_data_path and os.path.exists(daily_data_path):
        try:
            df = pd.read_csv(daily_data_path)
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.sort_values("date")
            day_mean = df.groupby(df["date"].dt.dayofyear)["rainfall"].mean()
            default = float(day_mean.loc[doy]) if doy in day_mean.index else float(day_mean.mean() if len(day_mean) else 1.0)
            # Ngày "hôm nay" đã ánh xạ vào lịch sử — MA tính trước ngày đó (không phải cuối file)
            today_hist = _map_calendar_to_history_date(pd.Timestamp(date_obj), df)
            for lag in [1, 2, 7, 14, 30]:
                prev = date_obj - timedelta(days=lag)
                prev_hist = _map_calendar_to_history_date(pd.Timestamp(prev), df)
                p = df[df["date"] == prev_hist]
                row[f"rainfall_lag_{lag}"] = float(p["rainfall"].iloc[0]) if len(p) > 0 else default
            for w in [3, 7, 14]:
                last = df[df["date"] < today_hist].tail(w)
                row[f"rainfall_ma_{w}"] = float(last["rainfall"].mean()) if len(last) >= 2 else default
            prev1_hist = _map_calendar_to_history_date(pd.Timestamp(date_obj - timedelta(days=1)), df)
            for col in ["temperature", "humidity", "wind_speed"]:
                if col in df.columns:
                    c_avg = df.groupby(df["date"].dt.dayofyear)[col].mean()
                    v = float(c_avg.loc[doy]) if doy in c_avg.index else float(df[col].mean())
                    row[f"{col}_lag_1"] = row[f"{col}_ma_7"] = v
                    p1 = df[df["date"] == prev1_hist]
                    if len(p1) > 0:
                        row[f"{col}_lag_1"] = float(p1[col].iloc[0])
                    last_t = df[df["date"] < today_hist].tail(7)
                    if len(last_t) >= 2 and col in last_t.columns:
                        row[f"{col}_ma_7"] = float(last_t[col].mean())
            if "cloud_cover" in df.columns:
                p1 = df[df["date"] == prev1_hist]
                row["cloud_cover_lag_1"] = float(p1["cloud_cover"].iloc[0]) if len(p1) > 0 else 50
            if "surface_pressure" in df.columns:
                p1 = df[df["date"] == prev1_hist]
                row["surface_pressure_lag_1"] = float(p1["surface_pressure"].iloc[0]) if len(p1) > 0 else 1013
        except Exception:
            pass
    row["temp_lag_1"] = row.get("temperature_lag_1", 0)
    row["temp_ma_7"] = row.get("temperature_ma_7", 0)
    for c in feature_cols:
        if c not in row:
            row[c] = 0
    return row


def _daily_features_from_history(daily_data_path, date_obj, feature_cols, base_year=1979):
    """Tạo dict feature cho một ngày (dùng chung cho mọi mô hình)."""
    return _build_daily_features_for_date(date_obj, daily_data_path, feature_cols, base_year)


def predict_rainfall_daily(model_path, year, month, day, daily_data_path=None):
    """Dự đoán lượng mưa theo NGÀY (mm/ngày) - mô hình regression đơn."""
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


def predict_rainfall_daily_two_stage(model_path, year, month, day, daily_data_path=None):
    """
    Dự đoán 2 giai đoạn: (1) Có mưa hay không (2) Nếu có thì lượng mưa bao nhiêu mm.
    Trả về: (has_rain: bool, amount_mm: float, metrics: dict) hoặc dict đầy đủ nếu cần rain_probability.
    """
    from datetime import date
    date_obj = date(int(year), int(month), int(day))
    if not os.path.exists(model_path):
        return False, 0.0, {}
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "classifier" not in data or "regressor" not in data:
        return False, 0.0, {}
    clf = data["classifier"]
    reg = data["regressor"]
    scaler = data.get("scaler")
    feature_cols = data.get("feature_cols", [])
    base_year = int(data.get("base_year", BASE_YEAR))
    feats = _build_daily_features_for_date(date_obj, daily_data_path, feature_cols, base_year)
    X = pd.DataFrame({c: [feats.get(c, 0)] for c in feature_cols})
    X_s = scaler.transform(X) if scaler else X.values
    has_rain = clf.predict(X_s)[0] == 1
    rain_prob = float(clf.predict_proba(X_s)[0][1]) if hasattr(clf, "predict_proba") else (1.0 if has_rain else 0.0)
    amount = float(reg.predict(X_s)[0]) if has_rain else 0.0
    amount = max(0.0, amount)
    metrics = dict(data.get("metrics", {}))
    metrics["rain_probability"] = rain_prob
    return has_rain, amount, metrics
