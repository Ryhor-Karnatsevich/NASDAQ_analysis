import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

is_test = False
if is_test:
    path = r"../../Data/Test Data/test_analysis.csv"
    output_path = r"../../Data/Test Data/test_eda.csv"
else:
    path = r"../../Data/Main Data/all_stocks_analysis.csv"
    output_path = r"../../Data/Main Data/all_stocks_eda.csv"

df = pd.read_csv(path)
df['Date'] = pd.to_datetime(df['Date'])
all_tickers = df['Ticker'].unique()

# 1 Market Level
###------------------------------------------------------------------------------
### GET TICKERS
###------------------------------------------------------------------------------
run_market_eda = True
run_stock_eda = True
corr_sample = 20

if run_market_eda:
    # Returns Distribution
    plt.figure(figsize=(10, 5))

    lower = df["Returns"].quantile(0.01)
    upper = df["Returns"].quantile(0.99)
    filtered_returns = df[(df["Returns"] >= lower) & (df["Returns"] <= upper)]["Returns"]

    filtered_returns.hist(bins=100, alpha=0.7, color='steelblue')
    plt.title(f"Market Returns (Range: {lower:.2f} to {upper:.2f})")
    plt.show()

    # Volatility Distribution
    plt.figure(figsize=(10, 5))

    lower = df["Volatility"].quantile(0.01)
    upper = df["Volatility"].quantile(0.99)
    filtered_volatility = df[(df["Volatility"] >= lower) & (df["Volatility"] <= upper)]["Volatility"]

    filtered_volatility.hist(bins=300, alpha=0.7, color='orange')
    plt.title(f"Market Volatility Distribution (Range: {lower:.2f} to {upper:.2f})")
    plt.show()

    # CORRELATION MATRIX
    sample_corr_tickers = np.random.choice(all_tickers, corr_sample)
    pivot = df[df['Ticker'].isin(sample_corr_tickers)].pivot(index="Date", columns="Ticker", values="Returns")
    corr_matrix = pivot.corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm')
    plt.colorbar()
    plt.title(f"Correlation Matrix (Sample of {corr_sample} tickers)")
    plt.show()

# 2 Stock Level
###------------------------------------------------------------------------------
### GET TICKERS
###------------------------------------------------------------------------------
if run_stock_eda:
    def get_tickers(all_tickers):
        user_input = input("Tickers: ").strip()
        if user_input:
            tickers = [t.strip() for t in user_input.split(',')]
            return tickers
        else:
            n_input = input("Number: ").strip()
            if n_input:
                n_random = int(n_input)
                return list(np.random.choice(all_tickers, n_random, replace=False))
            else:
                return []


    selected = get_tickers(all_tickers)


# --------------------------------------------------------------------------------

    #### Graphics
    def graphics(selected):
        for ticker in selected:
            df[df['Ticker'] == ticker].plot(x='Date', y="Close", title=ticker)
            plt.show()


    ### Histograms
    def histogram(selected):
        for ticker in selected:
            data = df[df['Ticker'] == ticker]
            data['Returns'].hist(bins=100, figsize=(10, 5))
            plt.title(f"Returns Distribution: {ticker}")
            plt.show()


    ### Volatility Clustering
    def clustering(selected):
        for ticker in selected:
            data = df[df['Ticker'] == ticker]
            data.plot(x='Date', y='Volatility', figsize=(10, 5), title=f"Volatility over time: {ticker}")
            plt.show()


    ### Rolling Mean vs Price
    def SMA(selected):
        for ticker in selected:
            data = df[df['Ticker'] == ticker]
            plt.figure(figsize=(10, 5))
            plt.plot(data['Date'], data['Close'], label='Price', alpha=0.5)
            plt.plot(data['Date'], data['SMA_10'], label='MA 10', color='red')
            plt.title(f"Price vs Moving Average: {ticker}")
            plt.legend()
            plt.show()


    ### ACF
    def ACF(selected):
        for ticker in selected:
            plot_acf(df[df['Ticker'] == ticker]["Returns"].dropna(), lags=30)
            plt.title(f"Market Memory Check (ACF): {ticker}")
            plt.show()


###---------------------------------------------------------------------------
### Running all functions
###---------------------------------------------------------------------------
    if selected:
        graphics(selected)
        histogram(selected)
        clustering(selected)
        SMA(selected)
        ACF(selected)
    else:
        print("ERROR OR WRONG INPUT")



