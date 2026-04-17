#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline: Lấy dữ liệu từ API -> Chuẩn hóa dữ liệu -> Train/Đánh giá mô hình.
Chạy từ thư mục DuBao:  python run_pipeline.py [--fetch]
"""
import os
import sys
import argparse

DUBAO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(DUBAO_DIR, "src")
DATA_DIR = os.path.join(DUBAO_DIR, "data")

def run_step(step_name, mod_name, func_name="main"):
    """Chạy một bước: import module trong DuBao/src và gọi hàm."""
    sys.path.insert(0, SRC_DIR)
    os.chdir(SRC_DIR)
    try:
        mod = __import__(mod_name)
        fn = getattr(mod, func_name, None)
        if not callable(fn):
            print(f"  Bỏ qua: {mod_name}.{func_name} không tồn tại.")
            return True
        print(f"\n--- {step_name} ---")
        fn()
        return True
    except Exception as e:
        print(f"  Lỗi: {e}")
        return False
    finally:
        os.chdir(DUBAO_DIR)


def get_daily_predictions():
    """
    Dự đoán lượng mưa ngày mai bằng 3 mô hình ML.
    Trả về: pred_rf, pred_lr, pred_xgb
    """
    import pandas as pd
    import pickle
    from datetime import datetime, timedelta

    # Load models
    models_dir = os.path.join(DUBAO_DIR, "models")
    models = {}

    for name in ['rf', 'xgb', 'lr']:
        model_path = os.path.join(models_dir, f'{name}_daily_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
        else:
            raise FileNotFoundError(f"Model {name} not found: {model_path}")

    # Load latest features
    data_dir = os.path.join(DUBAO_DIR, "data")
    df = pd.read_csv(os.path.join(data_dir, 'daily_features.csv'))

    # Lấy dòng cuối cùng (ngày gần nhất có đủ features)
    latest = df.iloc[-1:].copy()
    
    # Features cần thiết (giống như trong training)
    exclude = {'date', 'target', 'datetime', 'rainfall'}
    feature_cols = [c for c in df.columns if c not in exclude]
    X_pred = latest[feature_cols]

    # Predict
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X_pred)[0]
        predictions[name] = max(0, pred)  # Không âm

    return predictions['rf'], predictions['lr'], predictions['xgb']


def retrain_models():
    """
    Retrain tất cả 3 models với dữ liệu mới nhất.
    """
    print("🔄 Retraining models...")

    # Import train function
    sys.path.insert(0, SRC_DIR)
    try:
        from train_daily_models import train_daily_models
        train_daily_models()
        print("✅ Models retrained successfully")
    except Exception as e:
        print(f"❌ Retrain failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline dữ liệu và huấn luyện mô hình.")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Gọi Open-Meteo API lấy/cập nhật dữ liệu thời tiết (nhiệt độ, độ ẩm, gió, mây, áp suất)",
    )
    parser.add_argument(
        "--extend-rain",
        action="store_true",
        help="Mở rộng dữ liệu mưa theo ngày (raw_daily.csv) đến 2025 bằng Open-Meteo (precipitation_sum)",
    )
    args = parser.parse_args()

    print("Thư mục DuBao:", DUBAO_DIR)
    print("Thư mục data:", DATA_DIR)

    if args.fetch:
        ok = run_step("Bước 1: Gọi API thời tiết (Open-Meteo)", "fetch_weather_data")
        if not ok:
            print("Có thể bỏ qua nếu đã có file weather_daily.csv. Tiếp tục...")
    else:
        if not os.path.exists(os.path.join(DATA_DIR, "weather_daily.csv")):
            print("Chưa có weather_daily.csv. Chạy với --fetch để tải dữ liệu từ API:")
            print("  python run_pipeline.py --fetch")
        print("\n--- Bước 1: Bỏ qua fetch (dùng dữ liệu hiện có) ---")

    if args.extend_rain:
        ok = run_step("Bước 1b: Mở rộng dữ liệu mưa đến 2025 (Open-Meteo)", "fetch_rainfall_extend")
        if not ok:
            print("Không mở rộng được raw_daily.csv. Tiếp tục với dữ liệu hiện có...")

    ok = run_step("Bước 2: Chuẩn hóa dữ liệu (ngày->tháng, merge weather)", "prepare_data")
    if not ok:
        print("Kiểm tra: raw_daily.csv, weather_daily.csv trong DuBao/data/")
        sys.exit(1)

    ok = run_step("Bước 3: Train & đánh giá mô hình (GB, RF, SARIMA)", "evaluate_all_models", "evaluate_models")
    if not ok:
        sys.exit(1)

    ok = run_step("Bước 4: Train 3 mô hình 2 giai đoạn (GB, RF, Extra Trees)", "train_daily_two_stage")
    if not ok:
        run_step("Bước 4 (fallback): Train mô hình daily đơn", "train_daily_model")

    print("\n✅ Pipeline xong. Chạy web: từ thư mục gốc project:")
    print('   python manage.py runserver')
    print("   Mở http://localhost:8000/")


if __name__ == "__main__":
    main()
