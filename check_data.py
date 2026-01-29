import pandas as pd
df = pd.read_csv(r'd:\Du Bao Luong Mua\DuBao\data\daily_combined.csv')
print('Shape:', df.shape)
print('Columns:', len(df.columns))
print('Sample columns:', df.columns[:10].tolist())