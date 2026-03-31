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

def run_unified_backtest(results, verbose=True):
    all_metrics = [] # For average performances
    strat_returns_dict = {} # For correlation matrix

    for r in results:
        ticker = r["summary"]["Ticker"]
        returns = r["series"]["returns"] / 100
        vol = r["series"]["volatility"] / 100

        # [1] Creating "holding" strategy
        weight_fixed = pd.Series(1, index=returns.index)
        ret_fixed = weight_fixed * returns

        # [2] Volatility Scaling strategy (Risk Reduction)
        target_vol = 0.02
        position = (target_vol / vol).clip(0, 1)  # No leverages, 0-1x
        ret_scaling = position * returns

        # [3] Volatility Filter (binary switch)
        threshold = vol.rolling(window=50).mean()
        weight_filter = (vol < threshold).astype(float)  # binary test
        return_filter = weight_filter * returns

        # [4] SMA based strategy
        volatility_sma = vol.rolling(window=200).mean()  # SMA_20
        weight_sma = (volatility_sma / vol).clip(0, 1)
        return_4 = weight_sma * returns

        # [5] Vol Scaling + Trend Filter
        momentum = returns.rolling(20).sum()  # кумулятивная доходность за 20 дней
        trend_signal = (momentum > 0).astype(float)  # 1 binary test
        weight_trend_scaling = position * trend_signal
        ret_trend_scaling = weight_trend_scaling * returns

        strat_returns_dict[ticker] = ret_scaling # store for matrix

        strategies = {
            "Fixed (B&H)": (ret_fixed, weight_fixed),
            "Vol Scaling (TVS)": (ret_scaling, position),
            "Vol Switch": (return_filter, weight_filter),
            "Vol Smart (SMA)": (return_4, weight_sma),
            "Vol Scaling & Switch": (ret_trend_scaling, weight_trend_scaling) # Fixed unpacking error
        }

        # Dict to store equity and DD for plotting
        plot_data = {}

        for name, (strat_ret, weight) in strategies.items():
            equity = (1 + strat_ret).cumprod()
            metrics = calculate_metrics(strat_ret, equity)

            if metrics:
                all_metrics.append({
                    "Ticker": ticker,
                    "Strategy": name,
                    **metrics
                })
                plot_data[name] = {
                    "equity": equity,
                    "dd": equity / equity.cummax() - 1,
                    "weight": weight
                }

        # --- VISUALIZATION PER TICKER ---
        if verbose:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f"Backtest Analysis: {ticker}", fontsize=16)

            # Plot 1: Equity Curves
            for name, data in plot_data.items():
                axes[0, 0].plot(data["equity"], label=name, alpha=0.9)
            axes[0, 0].set_title("Equity Curves (Cumulative Return)")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Drawdowns
            for name, data in plot_data.items():
                axes[0, 1].fill_between(data["dd"].index, data["dd"], 0, alpha=0.3, label=name)
            axes[0, 1].set_title("Drawdowns")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Dynamic Weights (TVS vs Smart)
            axes[1, 0].plot(position, label="Vol Scaling (TVS)", alpha=0.6, color='blue')
            axes[1, 0].plot(weight_sma, label="Vol Smart (SMA)", alpha=0.6, color='green')
            axes[1, 0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title("Strategy Exposure (Position Size)")
            axes[1, 0].set_ylim(-0.1, 1.1)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Volatility Diagnostics
            axes[1, 1].plot(np.abs(returns), color='gray', alpha=0.2, label='Realized |Return|')
            axes[1, 1].plot(vol, color='red', alpha=0.8, label='GARCH Predicted Vol')
            axes[1, 1].set_title("Volatility Analysis")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()


    # Final Correlation Matrix after all tickers are processed
    df_corr = pd.DataFrame(strat_returns_dict).corr()
    print("\n" + "=" * 50)
    print("CORRELATION MATRIX")
    print("=" * 50)
    print(df_corr.round(2))

    return pd.DataFrame(all_metrics)

# Function to calculate metrics for backtests comparison
def calculate_metrics(returns_series, equity_series):

    returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns_series) == 0:
        return None

    total_return = equity_series.iloc[-1] - 1
    sharpe = (returns_series.mean() / (returns_series.std() + 1e-8)) * np.sqrt(252)
    max_dd = (equity_series / equity_series.cummax() - 1).min()
    annual_vol = returns_series.std() * np.sqrt(252)
    win_rate = (returns_series > 0).mean()

    return {
        "Total Return": total_return,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Annual Vol": annual_vol,
        "Win Rate": win_rate
    }




# --- EXECUTION ---
results_df = run_unified_backtest(results, verbose=True)

# --- FINAL SUMMARY TABLE ---
summary = results_df.groupby("Strategy").agg({
    "Total Return": "mean",
    "Sharpe": "mean",
    "Max Drawdown": "mean",
    "Annual Vol": "mean",
    "Win Rate": "mean"
}).sort_values("Sharpe", ascending=False)

print("\n" + "=" * 80)
print("AVERAGE PERFORMANCE ACROSS ALL TICKERS")
print("=" * 80)
print(summary.round(4))

# Performance Check
pivot = results_df.pivot(index='Ticker', columns='Strategy', values='Sharpe')
print("\n" + "=" * 80)
print("STRATEGY WIN RATES (Sharpe > Buy & Hold)")
for col in [c for c in pivot.columns if c != "Fixed (B&H)"]:
    wins = (pivot[col] > pivot["Fixed (B&H)"]).sum()
    total = len(pivot)
    print(f"{col}: {wins}/{total} tickers ({wins / total:.1%})")


