from src.preprocess import daily_to_monthly, merge_weather_rainfall
from src.train_model import train_model
from src.train_sarima_model import train_sarima_model
from src.predict import predict_rainfall
from src.fetch_weather_data import fetch_weather_data

# Bước 1: Chuyển dữ liệu ngày sang tháng
print("🔄 Đang chuyển dữ liệu từ ngày → tháng...")
daily_to_monthly(
    input_file="data/raw_daily.csv",
    output_file="data/monthly_rainfall.csv"
)

# Bước 2: Fetch dữ liệu thời tiết
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

# Bước 4: Train mô hình
print("\n🤖 Đang train mô hình dự đoán lượng mưa với dữ liệu thời tiết...")

# Chọn mô hình
print("\nChọn mô hình:")
print("1. Gradient Boosting với weather data (khuyến nghị)")
print("2. SARIMA (chỉ rainfall)")

choice = input("Chọn (1 hoặc 2, mặc định 1): ").strip()

if choice == "2":
    print("\n🌪️ Train SARIMA...")
    train_sarima_model(
        csv_path="data/monthly_rainfall.csv",
        model_path="models/rainfall_model.pkl"
    )
else:
    print("\n🌪️ Train Gradient Boosting với weather data...")
    train_model(
        csv_path="data/monthly_combined.csv",
        model_path="models/rainfall_model.pkl"
    )

# Bước 5: Dự đoán thử
print("\n📅 Dự đoán lượng mưa...")
try:
    year = int(input("Nhập năm (1979-2100): "))
    month = int(input("Nhập tháng (1-12): "))
    
    if not (1979 <= year <= 2100) or not (1 <= month <= 12):
        print("❌ Năm phải từ 1979-2100 và tháng từ 1-12!")
    else:
        if choice == "2":  # SARIMA
            pred = predict_rainfall("models/rainfall_model.pkl", year, month)
        else:  # Gradient Boosting with weather
            pred = predict_rainfall("models/rainfall_model.pkl", year, month, "data/monthly_combined.csv")
        print(f"🌧️ Lượng mưa dự đoán tháng {month}/{year}:", round(pred, 2), "mm")
except ValueError:
    print("❌ Vui lòng nhập số!")
