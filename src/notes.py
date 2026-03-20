
##### DATA CLEANING CODE #####


import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\\Users\God\Desktop\CV\Python Stock Analysis\90_days.csv")

## Initial Data Check
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())


## Data Preparation

# Null Test
df = df.dropna()
print(df.isnull().sum())

# Duplicate test
duplicates = df.duplicated().sum()
print("Duplicates:", duplicates)

# Logical Test
print("High < Low:", len(df[df["high"] < df["low"]]))
print("Negative volume:", len(df[df["volume"] < 0]))
print("Close outside range:", len(df[(df["close"] > df["high"]) | (df["close"] < df["low"])]))


# Date Standardazing
df["date"] = pd.to_datetime(df["date"])

# Data Export
df.to_csv("clean_data.csv", index=False)







##### AGGREGATING #####
dft = df.groupby("Ticker").agg(
    close = ('Close','mean'),
    open = ('Open','mean')
).reset_index()

dft['close'] = round(dft['close'],2)
dft['open'] = round(dft['open'],2)

print(dft.head())