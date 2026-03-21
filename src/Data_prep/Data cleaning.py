import pandas as pd

is_test = False
if is_test:
    path = r"../../Data/Test Data/test_data.csv"
    output_path = r"../../Data/Test Data/test_data_cleaned.csv"
else:
    path = r"../../Data/Main Data/all_stocks.csv"
    output_path = r"../../Data/Main Data/all_stocks_cleaned.csv"

df = pd.read_csv(path)

### Basic check
# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())
# print(df.columns)

### Data Cleaning

df["Date"] = pd.to_datetime(df["Date"])

print(df[df.isnull().any(axis=1)])
df = df.dropna()
print(df.isnull().sum())

dup = df.duplicated().sum()
print(dup)
df = df.drop_duplicates()

df = df[
    (df["High"]>= df["Low"]) &
    (df["High"] >= df["Close"]) &
    (df["Low"] <= df["Close"]) &
    (df["Volume"] >= 0)
]
print(len(df[df["High"] < df["Low"]]))
print(len(df[df["High"] < df["Close"]]))
print(len(df[df["Low"]>df["Close"]]))
print(len(df[df["Volume"] < 0]))


df = df[
    (df["High"] > 0) &
    (df["Low"] > 0) &
    (df["Close"] > 0) &
    (df["Open"] > 0)
]
print(len(df[df["Open"] == 0]))
print(len(df[df["Close"] == 0]))
print(len(df[df["High"] == 0]))
print(len(df[df["Low"] == 0]))

df = df.sort_values(["Ticker","Date"])
df = df.reset_index(drop=True)

df.to_csv(output_path, index=False)







