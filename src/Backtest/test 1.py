import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open("../../Data/Results/garch_results.pkl", "rb") as f:
    results = pickle.load(f)

# ============================================
# SIMPLE VOLATILITY SCALING STRATEGY
# ============================================

def volatility_scaling_strategy(returns, vol, target_vol=0.02):

    returns = pd.Series(returns)
    vol = pd.Series(vol)

    returns, vol = returns.align(vol, join="inner")

    vol = vol.replace(0, np.nan).bfill()

    vol = vol / 100

    position = target_vol / vol
    position = position.clip(0, 2)  # меньше риск

    strategy_returns = position.shift(1) * (returns / 100)

    return strategy_returns


# ============================================
# BACKTEST FUNCTION
# ============================================

def run_backtest(results):
    all_results = []

    for r in results:
        ticker = r["summary"]["Ticker"]
        data = r["series"]

        returns = data["returns"]
        vol = data["volatility"]

        strat_returns = volatility_scaling_strategy(returns, vol)

        # Performance metrics
        cumulative_return = (1 + strat_returns).cumprod()
        total_return = cumulative_return.iloc[-1] - 1

        sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-8) * np.sqrt(252)

        max_drawdown = (cumulative_return / cumulative_return.cummax() - 1).min()

        all_results.append({
            "Ticker": ticker,
            "Total Return": total_return,
            "Sharpe": sharpe,
            "Max Drawdown": max_drawdown
        })

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(cumulative_return, label="Strategy")
        plt.title(f"{ticker} Strategy Performance")
        plt.legend()
        plt.show()

    return pd.DataFrame(all_results)


# ============================================
# RUN
# ============================================

if __name__ == "__main__":

    # IMPORT FROM YOUR MODEL FILE

    results_df = run_backtest(results)

    print("\n=== BACKTEST RESULTS ===")
    print(results_df.sort_values("Sharpe", ascending=False).round(4))