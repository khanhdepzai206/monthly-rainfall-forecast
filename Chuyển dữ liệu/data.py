import pandas as pd

input_file = "Mua Da Nang_(1979-2022).xlsx"    # tên file xlsx
output_file = "Mua Da Nang_(1979-2022).csv"    # tên file csv muốn xuất

# Đọc file Excel
df = pd.read_excel(input_file)

# Xuất ra CSV
df.to_csv(output_file, index=False, encoding="utf-8")

print("✓ Đã chuyển xlsx → csv thành công!")
