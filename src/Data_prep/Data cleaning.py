import pandas as pd

is_test = True
if is_test:
    path = r"../../Data/test_data.csv"
    output_path = r"../../Data/test_data_cleaned.csv"
else:
    path = r"../../Data/all_stocks.csv"
    output_path = r"../../Data/all_stocks_cleaned.csv"

df = pd.read_csv(path)

### Basic check
# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())
# print(df.columns)

### Data Cleaning

print(df[df.isnull().any(axis=1)])
df = df.dropna()
print(df.isnull().sum())

dup = df.duplicated().sum()
print(dup)

df = df[
    (df["High"]>= df["Low"]) &
    (df["High"] >= df["Close"]) &
    (df["Low"] <= df["Close"])
]

print(len(df[df["High"] < df["Low"]]))
print(len(df[df["High"] < df["Close"]]))
print(len(df[df["Low"]>df["Close"]]))
print(len(df[df["Volume"] < 0]))





