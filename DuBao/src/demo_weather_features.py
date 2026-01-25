import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Load dữ liệu
df = pd.read_csv('../data/monthly_combined.csv')
print('📊 Dữ liệu mẫu:')
print(df.head())
print(f'\n📈 Tổng số records: {len(df)}')
print(f'📅 Khoảng thời gian: {df.year.min()}-{df.year.max()}')

# Feature engineering
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['rainfall_lag_1'] = df['rainfall'].shift(1)
df['rainfall_lag_12'] = df['rainfall'].shift(12)
df['rainfall_ma_3'] = df['rainfall'].rolling(window=3, min_periods=1).mean()
df['rainfall_ma_12'] = df['rainfall'].rolling(window=12, min_periods=1).mean()
df['trend'] = range(len(df))
df['quarter'] = (df['month'] - 1) // 3 + 1

# Weather features
df['temp_lag_1'] = df['temperature'].shift(1)
df['temp_lag_12'] = df['temperature'].shift(12)
df['temp_ma_3'] = df['temperature'].rolling(window=3, min_periods=1).mean()
df['humidity_lag_1'] = df['humidity'].shift(1)
df['humidity_lag_12'] = df['humidity'].shift(12)
df['humidity_ma_3'] = df['humidity'].rolling(window=3, min_periods=1).mean()
df['wind_lag_1'] = df['wind_speed'].shift(1)
df['wind_lag_12'] = df['wind_speed'].shift(12)
df['wind_ma_3'] = df['wind_speed'].rolling(window=3, min_periods=1).mean()

df = df.dropna()

# Tách features và target
feature_cols = [col for col in df.columns if col not in ['rainfall', 'year', 'month', 'date']]
feature_cols.extend(['year', 'month'])

X = df[feature_cols]
y = df['rainfall']  # TARGET: Lượng mưa

print(f'\n🎯 TARGET VARIABLE: rainfall (lượng mưa)')
print(f'🔍 FEATURES ({len(feature_cols)}):')
for i, feature in enumerate(feature_cols, 1):
    print(f'  {i}. {feature}')

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

print(f'\n✅ Mô hình đã train xong!')
print(f'📊 Train set: {len(y_train)} samples')
print(f'🧪 Test set: {len(y_test)} samples')
print(f'🎯 Dự đoán: {len(y_pred)} giá trị rainfall')

# Demo prediction
print(f'\n🌧️ Ví dụ dự đoán:')
sample_idx = 0
sample_features = X_test.iloc[sample_idx]
actual_rainfall = y_test.iloc[sample_idx]
predicted_rainfall = y_pred[sample_idx]

print(f'  Tháng: {int(sample_features["month"])}/{int(sample_features["year"])}')
print(f'  Nhiệt độ: {sample_features["temperature"]:.1f}°C')
print(f'  Độ ẩm: {sample_features["humidity"]:.1f}%')
print(f'  Tốc độ gió: {sample_features["wind_speed"]:.1f} km/h')
print(f'  Lượng mưa thực tế: {actual_rainfall:.1f} mm')
print(f'  Lượng mưa dự đoán: {predicted_rainfall:.1f} mm')
print(f'  Sai số: {abs(predicted_rainfall - actual_rainfall):.1f} mm')