import pandas as pd
import numpy as np

is_test = True
if is_test:
    path = r"../../Data/test_data_cleaned.csv"
    output_path = r"../../Data/test_analysis.csv"
else:
    path = r"../../Data/all_stocks_cleaned.csv"
    output_path = r"../../Data/all_stocks_analysis.csv"

df = pd.read_csv(path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date'])

grouped = df.groupby("Ticker")
### Returns
df["Returns"] = grouped["Close"].pct_change()
### Log Returns
df["Log_Returns"] = np.log(df["Close"]) - np.log(grouped["Close"].shift(1))
### Standard Deviation
N = 10
df["Volatility"] = grouped["Returns"].rolling(window=N).std().reset_index(0,drop=True)
### SMA's
df["SMA_10"] = grouped["Close"].rolling(window=10).mean().reset_index(0, drop=True)
df["SMA_50"] = grouped["Close"].rolling(window=50).mean().reset_index(0, drop=True)



print(df.tail())

# 5. Моментум (Momentum) - изменение цены за 10 дней
df["Momentum"] = grouped["Close"].pct_change(periods=10)

# 6. Изменение объема (Volume Change)
df["Vol_Change"] = grouped["Volume"].pct_change()

# 7. ТАРГЕТ (Цель обучения)
# Предсказываем доходность на СЛЕДУЮЩИЙ день
# Мы берем доходность и "сдвигаем" её вверх для текущей строки
df["Target"] = grouped["Returns"].shift(-1)

# --- ФИНАЛ ---

# Удаляем строки с NaN (они появятся в начале каждого тикера из-за сдвигов)
df = df.dropna()

# Сохраняем
df.to_csv(output_path, index=False)
print(f"Готово! Данные с признаками сохранены в {output_path}")
print(df.head())
