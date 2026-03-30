import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open("../../Data/Results/garch_results.pkl", "rb") as f:
    results = pickle.load(f)


# ============================================
# BACKTEST FUNCTION - COMPARE TWO STRATEGIES
# ============================================

def run_backtest_comparison(results):
    """
    Compares two strategies:
    1. Fixed position (constant exposure)
    2. Volatility scaling (dynamic position based on GARCH forecast)
    """

    all_results = []

    for r in results:
        ticker = r["summary"]["Ticker"]
        data = r["series"]

        returns = data["returns"] / 100  # Convert % to decimal
        vol = data["volatility"] / 100  # Convert % to decimal

        # Strategy 1: Fixed position (always invest 1)
        fixed_returns = returns

        # Strategy 2: Dynamic position based on volatility
        # Inverse volatility: higher volatility = smaller position
        target_vol = 0.02  # Target daily volatility 2%
        position = target_vol / vol
        position = position.clip(0, 2)  # Limit position to 0-2x
        scaling_returns = position * returns

        # Calculate metrics
        equity_fixed = (1 + fixed_returns).cumprod()
        equity_scaling = (1 + scaling_returns).cumprod()

        def calculate_metrics(returns_series, equity_series):
            returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns_series) == 0:
                return None

            total_return = equity_series.iloc[-1] - 1
            sharpe = returns_series.mean() / (returns_series.std() + 1e-8) * np.sqrt(252)
            max_dd = (equity_series / equity_series.cummax() - 1).min()
            annual_vol = returns_series.std() * np.sqrt(252)
            win_rate = (returns_series > 0).mean()

            return {
                "Total Return": total_return,
                "Sharpe Ratio": sharpe,
                "Max Drawdown": max_dd,
                "Annual Vol": annual_vol,
                "Win Rate": win_rate
            }

        metrics_fixed = calculate_metrics(fixed_returns, equity_fixed)
        metrics_scaling = calculate_metrics(scaling_returns, equity_scaling)

        if metrics_fixed and metrics_scaling:
            all_results.append({
                "Ticker": ticker,
                "Strategy": "Fixed",
                **metrics_fixed
            })

            all_results.append({
                "Ticker": ticker,
                "Strategy": "Volatility Scaling",
                **metrics_scaling
            })

            # Visualization
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))

            # Plot 1: Equity curves
            axes[0, 0].plot(equity_fixed, label='Fixed', alpha=0.8)
            axes[0, 0].plot(equity_scaling, label='Volatility Scaling', alpha=0.8)
            axes[0, 0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
            axes[0, 0].set_title(f'{ticker} - Equity Curves')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Drawdowns
            dd_fixed = equity_fixed / equity_fixed.cummax() - 1
            dd_scaling = equity_scaling / equity_scaling.cummax() - 1
            axes[0, 1].fill_between(dd_fixed.index, dd_fixed, 0, alpha=0.5, label='Fixed')
            axes[0, 1].fill_between(dd_scaling.index, dd_scaling, 0, alpha=0.5, label='Volatility Scaling')
            axes[0, 1].set_title(f'{ticker} - Drawdowns')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Position sizes
            position_plot = position.clip(0, 2)
            axes[1, 0].plot(position_plot, color='green', alpha=0.7)
            axes[1, 0].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Fixed = 1')
            axes[1, 0].set_title(f'{ticker} - Position Sizes (Volatility Scaling)')
            axes[1, 0].set_ylabel('Position (x)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Volatility vs |Return|
            axes[1, 1].plot(np.abs(returns), color='gray', alpha=0.3, label='|Return|')
            axes[1, 1].plot(vol, color='red', alpha=0.7, label='Predicted Vol')
            axes[1, 1].set_title(f'{ticker} - Volatility vs |Return|')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    results_df = pd.DataFrame(all_results)

    summary = results_df.groupby('Strategy').agg({
        'Total Return': ['mean', 'std'],
        'Sharpe Ratio': ['mean', 'std'],
        'Max Drawdown': ['mean', 'std'],
        'Annual Vol': ['mean', 'std'],
        'Win Rate': ['mean', 'std']
    }).round(4)

    return results_df, summary


# ============================================
# RUN BACKTEST
# ============================================

results_df, summary = run_backtest_comparison(results)

print("\n" + "=" * 60)
print("INDIVIDUAL RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(summary)

pivot = results_df.pivot(index='Ticker', columns='Strategy', values='Sharpe Ratio')
improved = (pivot['Volatility Scaling'] > pivot['Fixed']).sum()
total = len(pivot.dropna())
print(f"\nSharpe Improvement: {improved}/{total} tickers ({(improved / total) * 100:.1f}%)")

# results_df.to_csv("backtest_results.csv", index=False)
# summary.to_csv("backtest_summary.csv")

# ============================================
# RUN
# ============================================