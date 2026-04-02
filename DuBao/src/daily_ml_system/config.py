# -*- coding: utf-8 -*-
"""Đường dẫn và ngưỡng hệ thống dự đoán mưa ngày mai."""
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data")
MODEL_DIR = os.path.join(_ROOT, "models", "daily_ml_system")

DAILY_COMBINED = os.path.join(DATA_DIR, "daily_combined.csv")
PREDICTION_LOG = os.path.join(DATA_DIR, "prediction_log.csv")
MODEL_BUNDLE_PATH = os.path.join(MODEL_DIR, "triple_models.pkl")

# Ngưỡng MAE trung bình 7 ngày gần nhất — vượt quá thì retrain
MAE_THRESHOLD = 2.0
WINDOW_DAYS = 7

BASE_YEAR = 1979
