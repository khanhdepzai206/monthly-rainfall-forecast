#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hệ thống Machine Learning hoàn chỉnh cho dự đoán lượng mưa hàng ngày.
Tự động: Train → Predict → Update Actual → Evaluate → Auto Retrain
"""
import os
import sys
import argparse
from datetime import datetime

def run_initial_setup():
    """Chạy setup ban đầu: chuẩn bị dữ liệu và train models."""
    print("🚀 INITIAL SETUP")
    print("=" * 50)

    # 1. Chuẩn bị dữ liệu
    print("\n1. Chuẩn bị dữ liệu...")
    from prepare_daily_data import prepare_daily_data
    prepare_daily_data()

    # 2. Train models
    print("\n2. Train models...")
    from train_daily_models import train_daily_models
    train_daily_models()

    print("\n✅ Initial setup hoàn thành!")

def run_daily_prediction():
    """Chạy prediction cho ngày mai."""
    print("🔮 DAILY PREDICTION")
    print("=" * 50)

    from predict_daily_new import predict_daily
    predictions = predict_daily()

    if predictions:
        print("\n📊 Kết quả dự đoán ngày mai:")
        for model, pred in predictions.items():
            if pred is not None:
                print(f"  {model.upper()}: {pred:.2f} mm")
            else:
                print(f"  {model.upper()}: Lỗi")

def update_actual(date, rainfall):
    """Cập nhật actual rainfall."""
    print("📝 UPDATE ACTUAL RAINFALL")
    print("=" * 50)

    from update_actual import update_actual_rainfall
    success = update_actual_rainfall(date, rainfall)

    if success:
        # Chạy auto retrain
        print("\n🔄 Kiểm tra auto retrain...")
        from auto_retrain import check_and_retrain
        check_and_retrain()

        # Đánh giá models
        print("\n📊 Đánh giá models...")
        from evaluate_daily_models import evaluate_models
        evaluate_models()

def show_status():
    """Hiển thị trạng thái hệ thống."""
    print("📈 SYSTEM STATUS")
    print("=" * 50)

    # Chuyển đến thư mục DuBao
    dubao_dir = os.path.dirname(__file__)
    data_dir = os.path.join(dubao_dir, '..', 'data')
    models_dir = os.path.join(dubao_dir, '..', 'models')

    files_to_check = [
        os.path.join(data_dir, 'daily_features.csv'),
        os.path.join(data_dir, 'prediction_log.csv'),
        os.path.join(models_dir, 'rf_daily_model.pkl'),
        os.path.join(models_dir, 'xgb_daily_model.pkl'),
        os.path.join(models_dir, 'lr_daily_model.pkl'),
    ]

    print("📁 Files:")
    for file_path in files_to_check:
        exists = "✓" if os.path.exists(file_path) else "✗"
        filename = os.path.basename(file_path)
        print(f"  {exists} {filename}")

    # Hiển thị recent predictions
    log_path = os.path.join(data_dir, 'prediction_log.csv')
    if os.path.exists(log_path):
        import pandas as pd
        log_df = pd.read_csv(log_path)
        print(f"\n📊 Recent Predictions ({len(log_df)} total):")
        recent = log_df.tail(3)
        for _, row in recent.iterrows():
            date = row.get('date', row.get('target_date', 'N/A'))
            rf_pred = f"{row.get('rf_pred'):.1f}" if pd.notna(row.get('rf_pred')) else "N/A"
            actual = f"{row.get('actual'):.1f}" if pd.notna(row.get('actual')) else "Pending"
            print(f"  {date}: RF={rf_pred}mm, Actual={actual}mm")

    # Hiển thị best model
    eval_path = os.path.join(models_dir, 'model_evaluation.csv')
    if os.path.exists(eval_path):
        eval_df = pd.read_csv(eval_path)
        best_row = eval_df[eval_df['is_best'] == True]
        if not best_row.empty:
            best_model = best_row['model'].iloc[0].upper()
            best_error = best_row['mean_error'].iloc[0]
            print(f"  Best model: {best_model}, Mean error: {best_error:.4f} mm")

def main():
    parser = argparse.ArgumentParser(description="Daily Rainfall Prediction System")
    parser.add_argument('action', choices=['setup', 'predict', 'update', 'status'],
                       help='Action to perform')
    parser.add_argument('--date', help='Date for actual update (YYYY-MM-DD)')
    parser.add_argument('--rainfall', type=float, help='Actual rainfall amount (mm)')

    args = parser.parse_args()

    # Chuyển đến thư mục src
    os.chdir(os.path.dirname(__file__))

    if args.action == 'setup':
        run_initial_setup()

    elif args.action == 'predict':
        run_daily_prediction()

    elif args.action == 'update':
        if not args.date or args.rainfall is None:
            print("❌ Cần --date và --rainfall cho action update")
            print("Example: python daily_ml_system.py update --date 2024-01-15 --rainfall 5.2")
            sys.exit(1)
        update_actual(args.date, args.rainfall)

    elif args.action == 'status':
        show_status()

if __name__ == "__main__":
    print("🌧️ Daily Rainfall Prediction System")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    main()

    print("\n🎯 Done!")