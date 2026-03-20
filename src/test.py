import pandas as pd

df = pd.read_csv(r"../Data/all_stocks.csv")

print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])