#!/usr/bin/env python
import pandas as pd
import joblib
import os

# Replicate the Django prediction logic
models_dir = 'DuBao/models'
data_dir = 'DuBao/data'

# Load latest features
df = pd.read_csv(os.path.join(data_dir, 'daily_features.csv'))
latest_features = df.iloc[-1:].copy()

# Remove non-feature columns
for drop_col in ['date', 'datetime', 'rainfall', 'target']:
    if drop_col in latest_features.columns:
        latest_features = latest_features.drop(drop_col, axis=1)

feature_columns = [c for c in latest_features.columns]
print(f"Using {len(feature_columns)} features: {feature_columns[:5]}...")

features_for_prediction = latest_features[feature_columns]
print(f"Features shape: {features_for_prediction.shape}")
print(f"Features dtypes: {features_for_prediction.dtypes.unique()}")

# Load model
rf_model = joblib.load(os.path.join(models_dir, 'rf_daily_model.pkl'))
print(f"Model loaded: {type(rf_model)}")

# Try prediction
try:
    pred = rf_model.predict(features_for_prediction)
    print(f"✅ Prediction successful: {pred}")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    print(f"Expected features: {rf_model.feature_names_in_}")
    print(f"Provided features: {list(features_for_prediction.columns)}")