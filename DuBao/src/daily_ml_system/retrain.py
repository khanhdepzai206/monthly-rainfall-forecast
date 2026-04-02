# -*- coding: utf-8 -*-
"""Tự động retrain nếu MAE 7 ngày vượt ngưỡng."""
from typing import Any, Dict, List, Optional

from . import config
from .evaluate import mae_last_n_days
from .train import train_all


def check_and_retrain(
    threshold: Optional[float] = None,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Nếu bất kỳ mô hình nào có MAE trung bình 7 ngày > threshold thì gọi train_all().
    """
    threshold = threshold if threshold is not None else config.MAE_THRESHOLD
    mae_7, _ = mae_last_n_days(config.WINDOW_DAYS, log_path)
    need = [k for k, v in mae_7.items() if v == v and v > threshold]
    result: Dict[str, Any] = {
        "threshold": threshold,
        "mae_7": mae_7,
        "models_retrain_triggered": need,
        "retrained": False,
    }
    if need:
        print(f"Retrain do MAE vượt {threshold}: {need}")
        train_all()
        result["retrained"] = True
        result["model_path"] = config.MODEL_BUNDLE_PATH
    else:
        print("Không cần retrain (MAE trong ngưỡng hoặc chưa đủ dữ liệu).")
    return result


def main():
    check_and_retrain()


if __name__ == "__main__":
    main()
