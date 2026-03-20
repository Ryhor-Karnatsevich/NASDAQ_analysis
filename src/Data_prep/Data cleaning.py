import pandas as pd

df = pd.read_csv(r"../../Data/test_data.csv")

### Basic check
# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())
# print(df.columns)

### missing data
# print(df[df.isnull().any(axis=1)])
# df = df.dropna()
# print(df.isnull().sum())

dup = df.duplicated().sum()
print(dup)




