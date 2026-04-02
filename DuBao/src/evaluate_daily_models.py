#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Đánh giá và xác định mô hình tốt nhất dựa trên sai số.
"""
import pandas as pd
import numpy as np
import os

def evaluate_models():
    """Đánh giá performance của các models từ prediction log."""

    log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'prediction_log.csv')

    if not os.path.exists(log_path):
        print("❌ Không tìm thấy prediction log!")
        return None

    log_df = pd.read_csv(log_path)

    # Lọc các dòng có actual
    valid_df = log_df.dropna(subset=['actual'])

    if len(valid_df) == 0:
        print("⚠ Chưa có dữ liệu actual để đánh giá!")
        return None

    print(f"📊 Đánh giá dựa trên {len(valid_df)} predictions có actual")

    # Tính metrics cho từng model
    results = {}
    model_names = ['rf', 'xgb', 'lr']
    pred_cols = ['rf_pred', 'xgb_pred', 'lr_pred']
    error_cols = ['error_rf', 'error_xgb', 'error_lr']

    for name, pred_col, error_col in zip(model_names, pred_cols, error_cols):
        if pred_col in valid_df.columns and error_col in valid_df.columns:
            valid_preds = valid_df.dropna(subset=[pred_col, error_col])

            if len(valid_preds) > 0:
                mean_error = valid_preds[error_col].mean()
                median_error = valid_preds[error_col].median()
                max_error = valid_preds[error_col].max()
                accuracy = (valid_preds[error_col] <= 1.0).mean() * 100  # % predictions within 1mm

                results[name] = {
                    'mean_error': mean_error,
                    'median_error': median_error,
                    'max_error': max_error,
                    'accuracy_1mm': accuracy,
                    'sample_count': len(valid_preds)
                }

                print(f"\n{name.upper()} Model:")
                print(".4f")
                print(".4f")
                print(".4f")
                print(".1f")
                print(f"  Samples: {len(valid_preds)}")

    if not results:
        print("❌ Không có đủ dữ liệu để đánh giá!")
        return None

    # Tìm model tốt nhất (error thấp nhất)
    best_model = min(results.items(), key=lambda x: x[1]['mean_error'])

    print(f"\n🏆 **MODEL TỐT NHẤT: {best_model[0].upper()}**")
    print(".4f")
    print(".1f")

    # Lưu kết quả đánh giá
    eval_results = []
    for name, metrics in results.items():
        eval_results.append({
            'model': name,
            'mean_error': metrics['mean_error'],
            'median_error': metrics['median_error'],
            'max_error': metrics['max_error'],
            'accuracy_1mm': metrics['accuracy_1mm'],
            'sample_count': metrics['sample_count'],
            'is_best': name == best_model[0],
            'evaluated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    eval_df = pd.DataFrame(eval_results)
    eval_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_evaluation.csv')
    eval_df.to_csv(eval_path, index=False)
    print(f"\n✓ Đã lưu kết quả đánh giá: {eval_path}")

    return best_model[0], results

if __name__ == "__main__":
    evaluate_models()