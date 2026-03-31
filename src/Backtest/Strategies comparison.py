import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# --- Window settings ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.4f}'.format

# Import data
with open("../../Data/Results/garch_results.pkl", "rb") as f:
    results = pickle.load(f)


# Strategies Function
def strategies_backtest(results, verbose=True):
    all_metrics = [] # For average performances
    strat_returns_dict = {} # For corr matrix
    plot_data = {} # To store data for Plots

    # Begins iterating on shares [ STRATEGIES, METRICS, VISUALIZATION ]
    for r in results:
        ticker = r["summary"]["Ticker"]
        returns = r["series"]["returns"] / 100
        vol = r["series"]["volatility"] / 100


        ### STRATEGIES
        #----------------------------------------------------------------------------------
        # [1] "Buy & Hold" strategy
        weight_fixed = pd.Series(1, index=returns.index)
        return_1 = weight_fixed * returns

        # [2] Volatility Scaling strategy (Risk Reduction)
        target_vol = 0.02
        position = (target_vol / vol).clip(0, 1)  # No leverages, 0-1x
        return_2 = position * returns

        # [3] Volatility Filter (binary switch)
        threshold = vol.rolling(window=50).mean()
        weight_filter = (vol < threshold).astype(float)  # binary test
        return_3 = weight_filter * returns

        # [4] SMA based strategy
        volatility_sma = vol.rolling(window=20).mean()  # SMA_20
        weight_sma = (volatility_sma / vol).clip(0, 1)
        return_4 = weight_sma * returns

        # [5] Volatility Scaling + Momentum
        momentum_sum = returns.rolling(20).sum()  # cumulative return 20 days
        trend_signal = (momentum_sum > 0).astype(float)  # 1 binary test
        weight_trend_scaling = position * trend_signal # Get position form [2]
        return_5 = weight_trend_scaling * returns
        #----------------------------------------------------------------------------------



        ### METRICS
        # Store returns for further correlating matrix creating
        strat_returns_dict[ticker] = return_1

        # Dictionary with return and weight for final table
        strategies = {
            "Fixed (B&H)": (return_1, weight_fixed),
            "Vol Scaling (TVS)": (return_2, position),
            "Vol Switch": (return_3, weight_filter),
            "Vol Smart (SMA)": (return_4, weight_sma),
            "Vol Scaling & Switch": (return_5, weight_trend_scaling)
        }

        # Another subiteration to add values to final table and to plot dictionary
        for name, (strategy_return, weight) in strategies.items():
            equity = (1 + strategy_return).cumprod() # creates equity curve (line of returns of all previous time)
            metrics = calculate_metrics(strategy_return, equity) # Use function that below

            if metrics:
                # Adding data to list for final table
                all_metrics.append({
                    "Ticker": ticker,
                    "Strategy": name,
                    **metrics # from function "calculate_metrics"
                })
                # Adding data to list to create plots
                plot_data[name] = {
                    "equity": equity,
                    "drawdown": equity / equity.cummax() - 1,
                    "weight": weight
                }


        ### VISUALIZATION
        #------------------------------------------------------------------------------------------------
        if verbose:
            # Setup for one board with 4 plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f"Strategies Comparison for : {ticker}", fontsize=16)

            # Plot 1: Equity Curves
            for name, data in plot_data.items():
                axes[0, 0].plot(data["equity"], label=name, alpha=0.9)
            axes[0, 0].set_title("Equity Curves (Cumulative Return)")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Drawdowns
            for name, data in plot_data.items():
                axes[0, 1].fill_between(data["drawdown"].index, data["drawdown"], 0, alpha=0.3, label=name)
            axes[0, 1].set_title("Drawdowns")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Dynamic Weights (TVS vs Smart)
            axes[1, 0].plot(position, label="Vol Scaling (TVS)", alpha=0.6, color='C0')
            # axes[1, 0].plot(weight_filter, label="Vol Switch", alpha=0.6, color='C10')                    # not the best option
            axes[1, 0].plot(weight_sma, label="Vol Smart (SMA)", alpha=0.6, color='C2')
            # axes[1, 0].plot(weight_trend_scaling, label="Vol Scaling & Switch", alpha=0.6, color='C30')   # not the best option
            axes[1, 0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title("Position Size")
            axes[1, 0].set_ylim(-0.1, 1.1)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Volatility Diagnostics
            axes[1, 1].plot(np.abs(returns), color='gray', alpha=0.3, label='Realized |Return|')
            axes[1, 1].plot(vol, color='red', alpha=0.8, label='EGARCH Predicted Volatility')
            axes[1, 1].set_title("Volatility Prediction vs |Returns|")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
    ### END of iterations


    # Correlation Matrix
    #-----------------------------------------------------------------------------------------------
    # Created dataframe from dictionary with a ticker as a key and series as a value
    df_corr = pd.DataFrame(strat_returns_dict).corr()
    print("\n" + "=" * 80)
    print("CORRELATION MATRIX")
    print("=" * 80)
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(df_corr)

    # Mean correlation excluding the diagonal
    mean_corr = (df_corr.sum().sum() - len(df_corr)) / (len(df_corr) ** 2 - len(df_corr))
    print(f"Average Correlation: {mean_corr:.2f}")
    # -----------------------------------------------------------------------------------------------

    return pd.DataFrame(all_metrics)



# Function to calculate metrics
#----------------------------------------------------------------------------------------------------------------------------------
def calculate_metrics(returns_series, equity_series):

    # Data cleaning
    returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns_series) == 0:
        return None

    # Metrics
    total_return = equity_series.iloc[-1] - 1                                       # Return for the period
    sharpe = (returns_series.mean() / (returns_series.std() + 1e-8)) * np.sqrt(252) # Sharpe
    max_dd = (equity_series / equity_series.cummax() - 1).min()                     # Max Drawdown
    annual_vol = returns_series.std() * np.sqrt(252)                                # Annual Volatility
    hit_ratio = (returns_series > 0).mean()                                         # Percentage of Days when strategy return > 0

    # That dictionary appends to ticker dictionary in all_metrics list
    return {
        "Total Return": total_return,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Annual Vol": annual_vol,
        "Hit Ratio": hit_ratio
    }
#----------------------------------------------------------------------------------------------------------------------------------


# EXECUTION
results_df = strategies_backtest(results, verbose=False)


# FINAL SUMMARY TABLE PRINTING
summary = results_df.groupby("Strategy").agg({
    "Total Return": "mean",
    "Sharpe": "mean",
    "Max Drawdown": "mean",
    "Annual Vol": "mean",
    "Hit Ratio": "mean"
}).sort_values("Sharpe", ascending=False)

print("\n" + "=" * 80)
print("AVERAGE PERFORMANCE OF STRATEGIES")
print("=" * 80)
print("=" * 80)
print_summary = summary.copy()
print_summary["Hit Ratio"] = print_summary["Hit Ratio"].apply(lambda x: f"{x:.2f}")
print(print_summary)



# Performance Check
pivot = results_df.pivot(index='Ticker', columns='Strategy', values='Sharpe')
print("\n" + "=" * 80)
print("STRATEGY WIN RATES (Sharpe > Buy & Hold)")
for col in [c for c in pivot.columns if c != "Fixed (B&H)"]:
    wins = (pivot[col] > pivot["Fixed (B&H)"]).sum()
    total = len(pivot)
    print(f"{col}: {wins}/{total} tickers ({wins / total:.1%})")


