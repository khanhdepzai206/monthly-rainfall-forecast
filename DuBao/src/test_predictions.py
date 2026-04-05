#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test predictions on latest data.
"""
import pandas as pd
import joblib
import os

# Load latest features
data_dir = os.path.join('..', 'data')
df = pd.read_csv(os.path.join(data_dir, 'daily_features.csv'))
latest = df.iloc[-1:]

# Load models
models_dir = os.path.join('..', 'models')
rf_model = joblib.load(os.path.join(models_dir, 'rf_daily_model.pkl'))
xgb_model = joblib.load(os.path.join(models_dir, 'xgb_daily_model.pkl'))
lr_model = joblib.load(os.path.join(models_dir, 'lr_daily_model.pkl'))

# Get features
feature_cols = [c for c in latest.columns if c not in ['date', 'target', 'datetime', 'rainfall']]
features = latest[feature_cols]

# Predict
rf_pred = rf_model.predict(features)[0]
xgb_pred = xgb_model.predict(features)[0]
lr_pred = lr_model.predict(features)[0]

print(f'Latest predictions - RF: {rf_pred:.2f}, XGB: {xgb_pred:.2f}, LR: {lr_pred:.2f}')
print(f'Actual last rainfall: {df.iloc[-1]["rainfall"]:.2f}')
print(f'Next day target: {df.iloc[-1]["target"]:.2f}')
print(f'Features used: {len(feature_cols)}')