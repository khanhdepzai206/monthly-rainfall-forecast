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
    """Chạy một bước: import module và gọi hàm main."""
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
        if DUBAO_DIR:
            os.chdir(DUBAO_DIR)


def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline dữ liệu và huấn luyện mô hình.")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Gọi Open-Meteo API lấy/cập nhật dữ liệu thời tiết (nhiệt độ, độ ẩm, gió, mây, áp suất)",
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

    ok = run_step("Bước 2: Chuẩn hóa dữ liệu (ngày->tháng, merge weather)", "prepare_data")
    if not ok:
        print("Kiểm tra: raw_daily.csv, weather_daily.csv trong DuBao/data/")
        sys.exit(1)

    ok = run_step("Bước 3: Train & đánh giá mô hình (GB, RF, SARIMA)", "evaluate_all_models", "evaluate_models")
    if not ok:
        sys.exit(1)

    ok = run_step("Bước 4: Train 3 mô hình dự đoán theo ngày (GB, RF, LR)", "train_daily_model")
    if not ok:
        sys.exit(1)

    print("\n✅ Pipeline xong. Chạy web: từ thư mục gốc project:")
    print('   python manage.py runserver')
    print("   Mở http://localhost:8000/")


if __name__ == "__main__":
    main()
