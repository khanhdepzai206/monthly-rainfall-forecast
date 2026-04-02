# -*- coding: utf-8 -*-
"""Đánh giá sai số log; mô hình tốt nhất; MAE N ngày gần nhất."""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import config
from .log_io import read_log


def mae_last_n_days(
    n: int = None,
    log_path: Optional[str] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    MAE trung bình (absolute error) của n ngày gần nhất có đủ actual + error_*.
    Trả về ({ 'rf': ..., 'xgb': ..., 'lr': ... }, dataframe slice).
    """
    n = n or config.WINDOW_DAYS
    log = read_log(log_path)
    if log.empty or "actual" not in log.columns:
        return {"rf": float("nan"), "xgb": float("nan"), "lr": float("nan")}, log
    log = log.copy()
    log["target_date"] = pd.to_datetime(log["target_date"])
    sub = log.dropna(subset=["actual", "error_rf", "error_xgb", "error_lr"]).sort_values("target_date")
    if sub.empty:
        return {"rf": float("nan"), "xgb": float("nan"), "lr": float("nan")}, sub
    tail = sub.tail(n)
    out = {
        "rf": float(tail["error_rf"].mean()),
        "xgb": float(tail["error_xgb"].mean()),
        "lr": float(tail["error_lr"].mean()),
    }
    return out, tail


def best_model(mae_dict: Dict[str, float]) -> Optional[str]:
    """Tên mô hình có MAE thấp nhất (bỏ qua nan)."""
    valid = {k: v for k, v in mae_dict.items() if v == v and np.isfinite(v)}
    if not valid:
        return None
    return min(valid, key=valid.get)


def full_report(log_path: Optional[str] = None) -> Dict[str, Any]:
    mae_7, tail = mae_last_n_days(config.WINDOW_DAYS, log_path)
    best = best_model(mae_7)
    return {
        "mae_last_7_days": mae_7,
        "best_model": best,
        "best_model_name": {"rf": "RandomForest", "xgb": "XGBoost", "lr": "LinearRegression"}.get(
            best or "", best
        ),
        "rows_used": len(tail),
    }


def main():
    r = full_report()
    print("MAE (7 ngày gần nhất):", r["mae_last_7_days"])
    print("Mô hình tốt nhất:", r["best_model_name"], f"({r['best_model']})")


if __name__ == "__main__":
    main()
