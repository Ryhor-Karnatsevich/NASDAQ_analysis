import time
start_time = time.time()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed

# Additional configurations for pycharm console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.4f}'.format

#--------------------------------------------------------------------
path = r"../../Data/Main Data/all_stocks_analysis.csv"
df = pd.read_csv(path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"])
#Filter
min_obs = 500
df = df.groupby("Ticker").filter(lambda x: len(x) >= min_obs)
# Seed Setup
np.random.seed(52)


# Want to save results into csv file?
save = False
### SETUP
MODE = "FINAL"
split_date = "2019-01-01"
use_random = False
n = 100
# configurations for FINAL MODE
CONFIGS = {
    "DEFAULT": {"type": "EGARCH", "p": 2, "q": 1}
}

# setting up random list
if use_random:
    tickers = np.random.choice(df["Ticker"].unique(), size=n, replace=False)
else:
    tickers = ["MLPI"]
#["AAPL", "NVDA","GOOGL","MSFT","META","AMZN","MLPI"]

### GARCH Model
def garch_run(df,ticker,split_date,type,p,q,verbose=True):
#-----------------------------------------------------------
    data = df[df['Ticker'] == ticker].copy()
    data.set_index("Date", inplace=True)
    returns = data["Returns"].replace([np.inf, -np.inf], np.nan).dropna() * 100
    returns = returns.clip(-100, 100)

    train = returns[returns.index < split_date]
    test = returns[returns.index >= split_date]
    # Additional Length filtrating
    if len(train) < 500:
        print(f"{ticker} has less than 500 stocks")
        return None
#-----------------------------------------------------------
# Model creating
    # Training model
    model = arch_model(train, vol=type, p=p, q=q, dist='t',rescale = True)
    result = model.fit(disp=0)
    # Test if model correctly optimized data
    if result.convergence_flag != 0:
         print(f"{ticker} {type}({p},{q}) does not match (flag={result.convergence_flag})")
         return None
    # Scaling
    sc = getattr(result, 'scale', 1.0)
    # Testing model
    model_full = arch_model(returns * sc, vol=type, p=p, q=q, dist='t', rescale=False)
    test_res = model_full.fix(result.params)

    # Metrics
    test_sigma = test_res.conditional_volatility[test.index] / sc # prediction line
    # mae
    mae = mean_absolute_error(np.abs(test), test_sigma)
    # relative mae for comparison
    mean_abs_return = np.mean(np.abs(test))
    relative_mae = mae / mean_abs_return

# Parameters + persistence
# --------------------------------------------------------------------------------------------
    alpha_sum = 0
    beta_sum = 0

    for i in range(1, p + 1):
        alpha_sum += result.params.get(f"alpha[{i}]", 0)

    for i in range(1, q + 1):
        beta_sum += result.params.get(f"beta[{i}]", 0)

    alpha = result.params.get("alpha[1]", np.nan)
    beta = result.params.get("beta[1]", np.nan)
    #delta
    if type == "APARCH":
        delta = result.params.get("delta", np.nan)
    else:
        delta = np.nan
    # persistence
    if type == "EGARCH":
        persistence = beta_sum
    else:
        persistence = alpha_sum + beta_sum

    if persistence > 1.00001 and type != "EGARCH":
        print(f"Warning: {ticker} {type}({p},{q}) has persistence {persistence} >= 1")
# --------------------------------------------------------------------------------------------
# Other
    # Forecast 1 day in the future volatility
    final_forecast = test_res.forecast(horizon=1)
    future_vol = np.sqrt(final_forecast.variance.iloc[-1].values[0] / (sc**2))
    # Filter for invalid models
    if not np.isfinite(future_vol) or relative_mae > 10 or alpha > 100:
        print(f"Skipping {ticker}: Invalid results (MAE too high or INF)")
        return None
    # volatility + prediction graphics
    if verbose:
        plt.figure(figsize=(10, 4))
        plt.plot(np.abs(test), color='gray', alpha=0.3, label='Realized Volatility')
        plt.plot(test_sigma, color='red', label=f'{type} Forecast')
        plt.title(f"{ticker} | {type}({p},{q})")
        plt.legend()
        plt.show()


    return {
        "summary": {
            # Information
            "Ticker": ticker,
            "Model": type,
            "p": p,
            "q": q,
            # Quality
            "AIC": result.aic,
            "MAE": mae,
            "Relative MAE": relative_mae,
            # Parameters
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "persistence": persistence,
            # Other
            "tomorrow_volatility": future_vol,
            "converged": result.convergence_flag == 0,
            "train size": len(train),
            "test size": len(test)
        },
        # Part for model backtest
        "series": {
            "returns": test,
            "volatility": test_sigma
        }
    }

# I used Parallel calculation to reduce some run time. It is about 5-10% on a small sample.

# GRID mode is for comparing models and their parameters with each other on a stock sample.
if MODE == "GRID":
    tasks = [(ticker, v, p, q)
             for ticker in tickers
             for v in ["GARCH","APARCH","EGARCH"]
             for p, q in [(1, 1), (2, 1),(1,2)]]

    results = Parallel(n_jobs=-1)(
        delayed(garch_run)(df, ticker, split_date, type=v, p=p, q=q, verbose=False)
        for ticker, v, p, q in tasks
    )
    results = [r for r in results if r is not None]
    results_df = pd.DataFrame([r["summary"] for r in results])
    results_df = pd.DataFrame(results).sort_values(["Ticker", "AIC"])

# This mode is for calculating chosen model after GRID TEST.
# It best suitable for big samples of stocks and for testing the model.
elif MODE == "FINAL":
    tasks = [(ticker, CONFIGS.get(ticker, CONFIGS["DEFAULT"])["type"],
              CONFIGS.get(ticker, CONFIGS["DEFAULT"])["p"],
              CONFIGS.get(ticker, CONFIGS["DEFAULT"])["q"])
             for ticker in tickers]

    results = Parallel(n_jobs=-1)(
        delayed(garch_run)(df, ticker, split_date, type=v, p=p, q=q, verbose=True)
        for ticker, v, p, q in tasks
    )
    results = [r for r in results if r is not None]
    results_df = pd.DataFrame([r["summary"] for r in results])
else:
    print("WRONG MODE")


# Final row with averages
if MODE == "FINAL" and not results_df.empty:
    avg_row = results_df.mean(numeric_only=True)
    avg_row["Ticker"] = "AVERAGE"
    avg_row["Model"] = results_df["Model"].iloc[0]
    avg_row["p"] = results_df["p"].iloc[0]
    avg_row["q"] = results_df["q"].iloc[0]
    avg_row["converged"] = f"{results_df['converged'].mean() * 100:.1f}%"
    results_df = results_df.sort_values("Relative MAE")
    results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
    results_df["train size"] = results_df["train size"].round(0).astype(int)
    results_df["test size"] = results_df["test size"].round(0).astype(int)

print(results_df.round(4))

# Saving results to csv
if MODE == "GRID":
    output_path = r"../../Data/Results/GRID.csv"
else:
    output_path = r"../../Data/Results/FINAL.csv"
if save:
    results_df.to_csv(output_path, index=False)
# Adding time for optimization comparison
end_time = time.time()
run_time = end_time - start_time
print(f"\n Run Time : {run_time:.2f} seconds")



import pickle
with open("../../Data/Results/garch_results.pkl", "wb") as f:
    pickle.dump(results, f)