import pandas as pd

df = pd.read_csv(r"../../Data/Main Data/all_stocks.csv")
df['Date'] = pd.to_datetime(df["Date"])

export_path = r"../../Data/Test Data/test_data.csv"

print(df.info())
print(df.head())
print(df.tail())

filter = df[df["Date"].dt.year >= 2019].reset_index(drop=True)
filter.to_csv(export_path, index=False)
