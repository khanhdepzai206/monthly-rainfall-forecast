#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tự động kiểm tra sai số và retrain models nếu cần.
"""
import os
from update_actual import get_recent_errors
from train_daily_models import train_daily_models

def check_and_retrain(threshold=2.0, days=7):
    """
    Kiểm tra sai số 7 ngày gần nhất.
    Nếu > threshold thì retrain model đó.

    Args:
        threshold (float): Ngưỡng sai số để retrain
        days (int): Số ngày để tính trung bình
    """

    print(f"🔍 Kiểm tra sai số {days} ngày gần nhất (threshold: {threshold} mm)...")

    errors = get_recent_errors(days)
    if not errors:
        print("⚠ Không có đủ dữ liệu để kiểm tra!")
        return

    models_to_retrain = []
    for model_name, stats in errors.items():
        mean_error = stats['mean']
        median_error = stats['median']
        if median_error > threshold:
            models_to_retrain.append(model_name)
            print(f"⚠ Model {model_name.upper()} cần retrain (median_error: {median_error:.4f} > {threshold})")
        else:
            print(f"✓ Model {model_name.upper()} OK (mean: {mean_error:.4f}, median: {median_error:.4f})")

    if not models_to_retrain:
        print("🎉 Tất cả models đều hoạt động tốt!")
        return

    print(f"\n🔄 Retrain {len(models_to_retrain)} model(s)...")
    train_daily_models()
    print("🎯 Hoàn thành auto retrain!")

if __name__ == "__main__":
    check_and_retrain()