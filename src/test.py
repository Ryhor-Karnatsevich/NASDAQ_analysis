import pandas as pd

df = pd.read_csv(r"../Data/all_stocks.csv")

### Basic check
# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())
# print(df.columns)

### missing data
print(df[df.isnull().any(axis=1)])
df = df.dropna()
print(df.isnull().sum())

dup = df.duplicated().sum()
print(dup)

### Logic Test

df = df[
    (df["High"]>= df["Low"]) & (df["High"] >= df["Close"]) & (df["Low"] <= df["Close"])
]
print(len(df[df["High"] < df["Low"]]))
print(len(df[df["High"] < df["Close"]]))
print(len(df[df["Low"]>df["Close"]]))
print(len(df[df["Volume"] < 0]))