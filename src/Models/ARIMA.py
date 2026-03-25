import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

is_test = False

if is_test:
    path = r"../../Data/Test Data/test_analysis.csv"
else:
    path = r"../../Data/Main Data/all_stocks_analysis.csv"

df = pd.read_csv(path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"])
#Filter
min_obs = 1000
df = df.groupby("Ticker").filter(lambda x: len(x) >= min_obs)
# Seed Setup
np.random.seed(52)


### ARIMA SETUP
split_date = "2019-01-01"
order = (1,0,0)
use_random = True
n = 10

if use_random:
    tickers = np.random.choice(df["Ticker"].unique(), size=n, replace=False)
else:
    tickers = ["AAPL", "NVDA"]

### Model Function
def model_run(df,ticker,order,split_date, verbose=True):
    data = df[df['Ticker'] == ticker].copy()
    data = data.sort_values("Date")
    data = data.set_index("Date")

    data = data.asfreq("B")
    data = data.ffill()
#-----------------------------------------------------------
    train = data[data.index < split_date]["Returns"]
    test = data[data.index >= split_date]["Returns"]
    print(f"Ticker: {ticker}")
    print("Train size:", len(train))
    print("Test size:", len(test))
# Additional Length filtrating
    if len(train) < 750:
        return None
#-----------------------------------------------------------
    model  = ARIMA(train,  order=order)
    result = model.fit()
    if verbose:
        print(result.summary())

# Forecast
    forecast = result.forecast(steps=len(test))

# METRICS
    mse = mean_squared_error(test, forecast)
    direction = np.mean(np.sign(forecast) == np.sign(test))

    print("MSE:", mse)
    print("Direction:", direction)

    return {
        "ticker": ticker,
        "ar_coef": result.params.get("ar.L1", np.nan),
        "p_value": result.pvalues.get("ar.L1", np.nan),
        "mse": mse,
        "direction": direction,
        "train size": len(train),
        "test size": len(test)

    }

#---------------------------------------------------------------
results = []

for ticker in tickers:
    r = model_run(df, ticker, order,split_date)
    if r is not None:
        results.append(r)

results_df = pd.DataFrame(results)

# Adding average row
#--------------------------------------------------------------------------------------
avg_row = results_df.mean(numeric_only=True)
avg_row["ticker"] = "AVG"
results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
#--------------------------------------------------------------------------------------

print(results_df.round(4))


