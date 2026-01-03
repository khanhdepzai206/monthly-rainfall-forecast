from src.preprocess import daily_to_monthly
from src.train_model import train_model
from src.train_sarima_model import train_sarima_model
from src.predict import predict_rainfall

# Bước 1: Chuyển dữ liệu ngày sang tháng
print("🔄 Đang chuyển dữ liệu từ ngày → tháng...")
daily_to_monthly(
    input_file="data/raw_daily.csv",
    output_file="data/monthly_rainfall.csv"
)

# Bước 2: Train mô hình
print("\n🤖 Đang train mô hình dự đoán lượng mưa...")

# Chọn mô hình
print("\nChọn mô hình:")
print("1. Gradient Boosting (nhanh, đơn giản)")
print("2. SARIMA (tốt cho dữ liệu mùa vụ)")

choice = input("Chọn (1 hoặc 2, mặc định 1): ").strip()

if choice == "2":
    print("\n🌪️ Train SARIMA...")
    train_sarima_model(
        csv_path="data/monthly_rainfall.csv",
        model_path="models/rainfall_model.pkl"
    )
else:
    print("\n🌪️ Train Gradient Boosting...")
    train_model(
        csv_path="data/monthly_rainfall.csv",
        model_path="models/rainfall_model.pkl"
    )

# Bước 3: Dự đoán thử
print("\n📅 Dự đoán lượng mưa...")
try:
    year = int(input("Nhập năm (1979-2100): "))
    month = int(input("Nhập tháng (1-12): "))
    
    if not (1979 <= year <= 2100) or not (1 <= month <= 12):
        print("❌ Năm phải từ 1979-2100 và tháng từ 1-12!")
    else:
        pred = predict_rainfall("models/rainfall_model.pkl", year, month)
        print(f"🌧️ Lượng mưa dự đoán tháng {month}/{year}:", round(pred, 2), "mm")
except ValueError:
    print("❌ Vui lòng nhập số!")
