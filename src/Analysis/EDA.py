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

### Records Distribution
def data_availability(df):
    counts = df.groupby("Ticker")["Date"].count()
    print("Observations per ticker:")
    print(counts.describe())
    counts.hist(bins=50)
    plt.title("Distribution of Observations per Ticker")
    plt.show()
data_availability(df)

# Filtration
min_obs = 500
df = df.groupby("Ticker").filter(lambda x: len(x) >= min_obs)
all_tickers = df['Ticker'].unique()


### EDA Setup
run_market_eda = True
corr_sample = 20

# 1 Market Level
###------------------------------------------------------------------------------
### GET TICKERS
###------------------------------------------------------------------------------
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

#### Stock Graphics
def price(selected):
    for ticker in selected:
        df.loc[df['Ticker'] == ticker].plot(x='Date', y="Close", title=ticker)
        plt.show()

### Returns distribution
def returns(selected):
    for ticker in selected:
        data = df.loc[df['Ticker'] == ticker].copy()
        data['Returns'].hist(bins=100, figsize=(10, 5))
        plt.title(f"Returns Distribution: {ticker}")
        plt.show()

### Volatility Clustering Check
def volatility(selected):
    for ticker in selected:
        data = df.loc[df['Ticker'] == ticker]
        data.plot(x='Date', y='Volatility', figsize=(10, 5), title=f"Volatility over time: {ticker}")
        plt.show()

### Moving Average vs Price
def SMA(selected):
    for ticker in selected:
        data = df.loc[df['Ticker'] == ticker]
        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Close'], label='Price', alpha=0.5)
        plt.plot(data['Date'], data['SMA_10'], label='MA 10', color='red')
        plt.title(f"Price vs Moving Average: {ticker}")
        plt.legend()
        plt.show()

### Autocorrelation Function
def ACF(selected, lags=30):
    for ticker in selected:
        ticker_data = df.loc[df['Ticker'] == ticker, "Returns"].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_acf(ticker_data, lags=lags, ax=ax)
        ax.set_ylim(-0.25, 0.25)
        plt.title(f"ACF: {ticker} (Range from -0.25 to 0.25)")
        plt.show()

###---------------------------------------------------------------------------
### Running all functions
###---------------------------------------------------------------------------
if selected:
    price(selected)
    returns(selected)
    volatility(selected)
    SMA(selected)
    ACF(selected)
else:
    print("ERROR OR WRONG INPUT")