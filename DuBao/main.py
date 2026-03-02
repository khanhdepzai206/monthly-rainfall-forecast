import os
import sys
import argparse

from src.preprocess import daily_to_monthly, merge_weather_rainfall, create_daily_combined
from src.train_model import train_model
from src.train_sarima_model import train_sarima_model
from src.train_two_step_daily import train_two_step_daily_model
from src.train_compare_models import train_compare_models
from src.predict import predict_rainfall
from src.predict_daily import predict_daily_rainfall, predict_daily_range
from src.predict_best_model import predict_with_best_model
from src.fetch_weather_data import fetch_weather_data

def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline dự đoán mưa")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Bắt buộc train lại mô hình (mặc định: tải mô hình đã lưu nếu tồn tại)"
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch dữ liệu thời tiết từ API"
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Dự đoán theo ngày thay vì theo tháng"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train và so sánh 2-3 mô hình khác nhau"
    )
    args = parser.parse_args()
    
    # Bước 1: Chuyển dữ liệu ngày sang tháng
    print("🔄 Đang chuyển dữ liệu từ ngày → tháng...")
    daily_to_monthly(
        input_file="data/raw_daily.csv",
        output_file="data/monthly_rainfall.csv"
    )

    # Bước 1b: Chuẩn bị dữ liệu ngày (nếu dùng --daily)
    if args.daily:
        print("\n📊 Đang chuẩn bị dữ liệu theo ngày...")
        create_daily_combined(
            raw_daily_path="data/raw_daily.csv",
            weather_daily_path="data/weather_daily.csv",
            output_file="data/daily_combined.csv"
        )

    # Bước 2: Fetch dữ liệu thời tiết (nếu --fetch)
    if args.fetch:
        print("\n🌤️ Đang fetch dữ liệu thời tiết từ Open-Meteo API...")
        try:
            fetch_weather_data("1979-01-01", "2022-12-31")
            print("✅ Đã fetch dữ liệu thời tiết thành công!")
        except Exception as e:
            print(f"❌ Lỗi khi fetch dữ liệu thời tiết: {e}")

    # Bước 3: Merge dữ liệu mưa và thời tiết
    print("\n🔗 Đang merge dữ liệu mưa và thời tiết...")
    merge_weather_rainfall(
        rainfall_file="data/monthly_rainfall.csv",
        weather_file="data/weather_daily.csv",
        output_file="data/monthly_combined.csv"
    )

    # ===== CHƯƠNG TRÌNH CHÍNH =====
    if args.compare:
        # ===== SO SÁNH NHIỀU MÔ HÌNH =====
        print("\n🔬 CHẾ ĐỘ SO SÁNH MÔ HÌNH")
        print("=" * 50)
        print("\n📊 Huấn luyện và so sánh các mô hình:")
        print("  1️⃣ GradientBoosting (hiện tại)")
        print("  2️⃣ RandomForest")
        print("  3️⃣ XGBoost (nếu có cài)")
        
        train_compare_models(
            csv_path="data/daily_combined.csv",
            output_dir="models"
        )
        
        print("\n" + "=" * 50)
        print("✅ Hoàn tất! Tìm thấy mô hình tốt nhất và lưu trong:")
        print("   - models/comparison_results.pkl")
        print("   - models/daily_classifier_*.pkl")
        print("   - models/daily_regressor_*.pkl")
        print("\n💡 Gợi ý: Dùng --daily để dự đoán với mô hình tốt nhất")
    
    elif args.daily:
        # ===== DỰ ĐOÁN THEO NGÀY (2-STEP) =====
        print("\n🌞 CHẾ ĐỘ DỰ ĐOÁN THEO NGÀY")
        print("=" * 50)
        
        classifier_path = "models/daily_classifier.pkl"
        regressor_path = "models/daily_regressor.pkl"
        
        # Kiểm tra dùng mô hình tốt nhất hay mặc định
        use_best_model = False
        try:
            import pickle
            with open("models/comparison_results.pkl", 'rb') as f:
                comparison = pickle.load(f)
            use_best_model = True
            print("✅ Tìm thấy kết quả so sánh, sẽ dùng mô hình tốt nhất")
            print(f"   Classifier tốt nhất: {comparison['best_classifier']}")
            print(f"   Regressor tốt nhất: {comparison['best_regressor']}")
        except:
            pass
        
        if use_best_model:
            print(f"✅ Tìm thấy mô hình đã lưu")
            print("📂 Sử dụng mô hình tốt nhất từ so sánh")
        elif os.path.exists(classifier_path) and os.path.exists(regressor_path) and not args.retrain:
            print(f"✅ Tìm thấy mô hình đã lưu")
            print("📂 Sử dụng mô hình hiện có (thêm --retrain để train lại)")
        else:
            if args.retrain:
                print("🔄 --retrain được kích hoạt, train lại mô hình...")
            else:
                print(f"⚠️ Không tìm thấy mô hình, train mô hình mới...")
            
            print("\n🎯 Huấn luyện 2 mô hình:")
            print("  1️⃣ Classifier: Dự đoán có mưa hay không")
            print("  2️⃣ Regressor: Dự đoán lượng mưa (nếu có mưa)")
            
            train_two_step_daily_model(
                csv_path="data/daily_combined.csv",
                classifier_path=classifier_path,
                regressor_path=regressor_path
            )
        
        print("\n✅ Mô hình sẵn sàng!")
        
        # Dự đoán thử
        print("\n📅 Dự đoán lượng mưa theo ngày...")
        try:
            year = int(input("Nhập năm (ví dụ 2023): "))
            month = int(input("Nhập tháng (1-12): "))
            day = int(input("Nhập ngày (1-31): "))
            
            if not (1 <= month <= 12) or not (1 <= day <= 31):
                print("❌ Tháng phải từ 1-12 và ngày từ 1-31!")
            else:
                if use_best_model:
                    result = predict_with_best_model(
                        csv_path="data/daily_combined.csv",
                        year=year,
                        month=month,
                        day=day
                    )
                    print(f"\n📅 Dự đoán ngày {day}/{month}/{year}:")
                    print(f"  🎯 Classifier: {result['classifier_model']}")
                    print(f"  🎯 Regressor: {result['regressor_model']}")
                else:
                    result = predict_daily_rainfall(
                        classifier_path=classifier_path,
                        regressor_path=regressor_path,
                        csv_path="data/daily_combined.csv",
                        year=year,
                        month=month,
                        day=day
                    )
                    print(f"\n📅 Dự đoán ngày {day}/{month}/{year}:")
                
                print(f"  🌦️ Có mưa: {'Có' if result['has_rain'] else 'Không'}")
                print(f"  📊 Xác suất mưa: {result['rain_probability'] * 100:.1f}%")
                print(f"  🌧️ Lượng mưa dự đoán: {result['predicted_rainfall']:.2f} mm")
        except ValueError:
            print("❌ Vui lòng nhập số!")
    
    else:
        # ===== DỰ ĐOÁN THEO THÁNG (MẶC ĐỊNH) =====
        print("\n📅 CHẾ ĐỘ DỰ ĐOÁN THEO THÁNG")
        print("=" * 50)
        
        # Bước 4: Train mô hình (hoặc tải mô hình đã lưu)
        print("\n🤖 Đang xử lý mô hình dự đoán lượng mưa...")
        
        model_path = "models/rainfall_model.pkl"
        
        # Kiểm tra mô hình đã tồn tại
        if os.path.exists(model_path) and not args.retrain:
            print(f"✅ Tìm thấy mô hình đã lưu: {model_path}")
            print("📂 Sử dụng mô hình hiện có (thêm --retrain để train lại)")
        else:
            if args.retrain:
                print("🔄 --retrain được kích hoạt, train lại mô hình...")
            else:
                print(f"⚠️ Không tìm thấy mô hình, train mô hình mới...")
            
            # Chọn mô hình
            print("\nChọn mô hình:")
            print("1. Gradient Boosting với weather data (khuyến nghị)")
            print("2. SARIMA (chỉ rainfall)")

            choice = input("Chọn (1 hoặc 2, mặc định 1): ").strip()

            if choice == "2":
                print("\n🌪️ Train SARIMA...")
                train_sarima_model(
                    csv_path="data/monthly_rainfall.csv",
                    model_path=model_path
                )
            else:
                print("\n🌪️ Train Gradient Boosting với weather data...")
                train_model(
                    csv_path="data/monthly_combined.csv",
                    model_path=model_path
                )
        
        print("\n✅ Mô hình sẵn sàng!")

        # Bước 5: Dự đoán thử
        print("\n📅 Dự đoán lượng mưa...")
        try:
            year = int(input("Nhập năm (1979-2100): "))
            month = int(input("Nhập tháng (1-12): "))
            
            if not (1979 <= year <= 2100) or not (1 <= month <= 12):
                print("❌ Năm phải từ 1979-2100 và tháng từ 1-12!")
            else:
                pred = predict_rainfall(model_path, year, month, "data/monthly_combined.csv")
                print(f"🌧️ Lượng mưa dự đoán tháng {month}/{year}:", round(pred, 2), "mm")
        except ValueError:
            print("❌ Vui lòng nhập số!")

if __name__ == "__main__":
    main()
