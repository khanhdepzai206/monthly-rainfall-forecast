# -*- coding: utf-8 -*-
"""
Tùy chọn: lưu log vào SQLite (prediction_log.db) — giống cột CSV.
Bật bằng biến môi trường DAILY_ML_USE_DB=1.
"""
import os
import sqlite3
from typing import Any, Dict, Optional

import pandas as pd

from . import config
from .log_io import COLUMNS

DB_PATH = os.path.join(os.path.dirname(config.PREDICTION_LOG), "prediction_log.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    c = _conn()
    c.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
        target_date TEXT PRIMARY KEY,
        prediction_date TEXT,
        rf_pred REAL,
        xgb_pred REAL,
        lr_pred REAL,
        actual REAL,
        error_rf REAL,
        error_xgb REAL,
        error_lr REAL
    )"""
    )
    c.commit()
    c.close()


def sync_from_csv():
    """Đồng bộ CSV -> DB (nếu dùng DB)."""
    if not os.path.exists(config.PREDICTION_LOG):
        return
    init_db()
    df = pd.read_csv(config.PREDICTION_LOG)
    c = _conn()
    for _, row in df.iterrows():
        c.execute(
            """INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                str(row.get("target_date", "")),
                str(row.get("prediction_date", "")),
                row.get("rf_pred"),
                row.get("xgb_pred"),
                row.get("lr_pred"),
                row.get("actual"),
                row.get("error_rf"),
                row.get("error_xgb"),
                row.get("error_lr"),
            ),
        )
    c.commit()
    c.close()
