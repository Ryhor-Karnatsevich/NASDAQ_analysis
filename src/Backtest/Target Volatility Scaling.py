import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import polars as pl

# Window settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.4f}'.format

# Import model
with open("../../Data/Results/garch_results.pkl", "rb") as f:
    results = pickle.load(f)

# Extract unique periods and create mapping
periods = sorted(set(r["summary"]["Period"] for r in results))
print(f"Periods found: {periods}")
target_periods = ['2007-06-01','2015-06-01','2018-06-01']

ticker_list = list(set(r["summary"]["Ticker"] for r in results))

# =====================================================
# Fast csv import and filtration to add close column
path = r"../../Data/Main Data/all_stocks_analysis.csv"
df = (
    pl.scan_csv(path)
    .select(["Date", "Ticker", "Close"])
    .filter(
        (pl.col("Ticker").is_in(ticker_list)) &
        (pl.col("Date") > "2015-01-01")
    )
    .collect()
)
df = df.to_pandas()
prices_pivot = df.pivot(index="Date", columns="Ticker", values="Close")
prices_pivot.index = pd.to_datetime(prices_pivot.index)
# =====================================================


def strategies_backtest(results, period_filter=None, verbose=False,
                        rebalance=0.035, vol_discount=0.8):

    all_metrics = []

    if period_filter:
        results = [r for r in results if r["summary"]["Period"] == period_filter]

    for r in results:
        ticker = r["summary"]["Ticker"]
        returns = r["series"]["returns"] / 100
        volatility = r["series"]["volatility"] / 100
        period = r["summary"]["Period"]

        # cutting test to 2 years
        start_dt = pd.to_datetime(period)
        end_dt = start_dt + pd.DateOffset(years=2)
        returns = returns[returns.index <= end_dt]
        volatility = volatility[volatility.index <= end_dt]

        # [1] "Buy & Hold" strategy
        fixed_fee = 0.0005
        weight_fixed = pd.Series(1, index=returns.index)
        return_fixed = weight_fixed * returns
        return_fixed.iloc[0] -= fixed_fee
        return_fixed.iloc[-1] -= fixed_fee

        # [2] "TVS with transaction and margin costs"
        # Costs Setup
        target_0 = 0.02
        margin_annual_rate_0 = 0.05  # 5% annually for leverage
        margin_daily_rate_0 = margin_annual_rate_0 / 252
        comm_rate_0 = 0.0005  # 0.05% - transaction cost
        # Raw return
        position_basic = (target_0 / volatility).clip(0, 2)
        return_raw_basic = position_basic * returns
        # Transaction Costs
        trades_basic = position_basic.diff().abs().fillna(0)
        commissions_0 = trades_basic * comm_rate_0
        # Margin Costs
        leverage_used_0 = (position_basic - 1).clip(lower=0)
        margin_costs_0 = leverage_used_0 * margin_daily_rate_0
        # Final Return
        return_net_basic = return_raw_basic - commissions_0 - margin_costs_0

        # ====================================================================================================
        # ====================================================================================================
        # ====================================================================================================

        # --------------------------------------------------
        # 1  Setup: Leverage, Costs and Prices
        leverage = 2
        vol_discount = 0.8  # value to offset predicted volatility, if it is overrated
        rebalance = 0.035  # value for rebalancing threshold
        comm_rate = 0.0005
        margin_daily_rate = 0.05 / 252  # annual rate / trading days
        ticker_prices = prices_pivot[ticker].reindex(returns.index)
        volatility = volatility.clip(lower=0.0125)  # limit flour for volatility

        # 2  Asymmetric Target (based on sma_200)
        sma200 = ticker_prices.rolling(200).mean().shift(1)
        volatility_target = np.where(ticker_prices > sma200, 0.02, 0.01)

        # 3  Volatility Risk Premium Offset
        raw_position = (volatility_target / (volatility * vol_discount)).clip(0, leverage)

        # 4  Rebalancing Threshold (Cutting unnecessary turnover & slippage)
        position = raw_position.copy()
        for i in range(1, len(position)):
            # Only update position if the change is greater than 5%
            if abs(raw_position.iloc[i] - position.iloc[i - 1]) < rebalance:
                position.iloc[i] = position.iloc[i - 1]

        # 5  Drawdown Protection (Circuit Breaker)
        # Calculating equity and drawdown to adjust position risk
        temp_returns = position * returns
        equity_temp = (1 + (temp_returns)).cumprod()
        drawdown_temp = equity_temp / equity_temp.cummax() - 1
        risk_multiplier = np.where(drawdown_temp < -0.1, 0.5, 1.0)
        risk_multiplier = pd.Series(risk_multiplier, index=position.index).shift(1).fillna(1.0)
        position = position * risk_multiplier

        # 6  Returns & Execution Costs
        return_raw = position * returns
        trades = position.diff().abs().fillna(0)
        commissions = trades * comm_rate

        # 7 Margin Costs (Cost of Leverage)
        leverage_used = (position - 1).clip(lower=0)
        margin_costs = leverage_used * margin_daily_rate

        # 8 Final Net Return for Optimized Strategy
        return_net = return_raw - commissions - margin_costs

        # ====================================================================================================
        # ====================================================================================================
        # ====================================================================================================

        # Dictionary with return and weight for final table
        strategies = {
            "Buy & Hold": (return_fixed, weight_fixed),
            "TVS with transaction and margin costs": (return_net_basic, position_basic),
            "Target Volatility Scaling (TVS)": (return_net, position)
        }

        # Another subiteration to add values to final table and to plot dictionary
        for name, (strategy_return, weight) in strategies.items():
            equity = (1 + strategy_return).cumprod()  # creates equity curve (line of returns of all previous time)
            metrics = calculate_metrics(strategy_return, equity)  # Use function that below

            if metrics:
                # Adding data to list for final table
                all_metrics.append({
                    "Ticker": ticker,
                    "Period": period,
                    "Strategy": name,
                    **metrics # from function "calculate_metrics"
                })

    return pd.DataFrame(all_metrics)


# Function to calculate metrics
# ----------------------------------------------------------------------------------------------------------------------------------
def calculate_metrics(returns_series, equity_series):
    # Data cleaning
    returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns_series) == 0:
        return None

    # Metrics
    total_return = equity_series.iloc[-1] - 1  # Return for the period
    sharpe = (returns_series.mean() / (returns_series.std() + 1e-8)) * np.sqrt(252)  # Sharpe
    max_dd = (equity_series / equity_series.cummax() - 1).min()  # Max Drawdown
    annual_vol = returns_series.std() * np.sqrt(252)  # Annual Volatility
    hit_ratio = (returns_series > 0).mean()  # Percentage of Days when strategy return > 0

    # That dictionary appends to ticker dictionary in all_metrics list
    return {
        "Total Return": total_return,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Annual Vol": annual_vol,
        "Hit Ratio": hit_ratio
    }

# =========================================================
# EXECUTION
# =========================================================

all_results = []

rebalance_values = [0.02, 0.03, 0.035, 0.05]
vol_discount_values = [0.6, 0.8, 1.0]

sensitivity_results = []

for period in periods:

    print("\n" + "="*60)
    print(f"Processing Period: {period}")
    print("="*60)

    filtered_results = [r for r in results if r["summary"]["Period"] == period]

    results_df = strategies_backtest(filtered_results)
    results_df["Period"] = period
    all_results.append(results_df)

    for reb in rebalance_values:
        for vd in vol_discount_values:

            tmp_df = strategies_backtest(
                filtered_results,
                rebalance=reb,
                vol_discount=vd
            )

            if not tmp_df.empty:
                tmp_df["Period"] = period
                tmp_df["rebalance"] = reb
                tmp_df["vol_discount"] = vd
                sensitivity_results.append(tmp_df)

final_df = pd.concat(all_results)
sensitivity_df = pd.concat(sensitivity_results)

# =========================================================
# REGIME ANALYSIS
# =========================================================

regime_map = {
    target_periods[0]: "2007-2009",
    target_periods[1]: "2015-2017",
    target_periods[2]: "2018-2020"
}

def map_regime(period):
    return regime_map.get(period, "Other")
# сначала фильтр
regime_df = final_df[final_df["Period"].isin(target_periods)].copy()
# потом создаём колонку
regime_df["Regime"] = regime_df["Period"].apply(map_regime)

regime_summary = regime_df.groupby(["Regime", "Strategy"]).agg({
    "Total Return": "mean",
    "Sharpe": "mean",
    "Max Drawdown": "mean"
})
regime_summary = regime_summary.sort_values(
    by=["Regime", "Sharpe"],
    ascending=[True, False]
)

print("\n" + "="*103)
print("REGIME ANALYSIS".center(103))
print("="*103)
print(regime_summary)

# =========================================================
# SENSITIVITY PLOT
# =========================================================

pivot = sensitivity_df.groupby(
    ["rebalance", "vol_discount", "Strategy"]
)["Sharpe"].mean().reset_index()

pivot_tvs = pivot[pivot["Strategy"] == "Target Volatility Scaling (TVS)"]

pivot_table = pivot_tvs.pivot(
    index="rebalance",
    columns="vol_discount",
    values="Sharpe"
)

pivot_table.plot(figsize=(10,6))
plt.title("Sensitivity of Sharpe to Parameters")
plt.grid(True)
plt.show()

# =========================================================
# FINAL SUMMARY
# =========================================================

summary = final_df.groupby("Strategy").agg({
    "Total Return": "mean",
    "Sharpe": "mean",
    "Max Drawdown": "mean",
    "Annual Vol": "mean",
    "Hit Ratio": "mean"
}).sort_values("Sharpe", ascending=False)

# Adding Outperformance column
win_rates = {}

pivot = final_df.pivot_table(
    index='Ticker',
    columns='Strategy',
    values='Sharpe'
)

for strategy in summary.index:
    if strategy != "Buy & Hold":
        wins = (pivot[strategy] > pivot["Buy & Hold"]).sum()
        total = len(pivot)
        win_rates[strategy] = f"{wins}/{total} ({wins / total:.1%})"

summary["Outperformance (vs B&H)"] = summary.index.map(
    lambda x: win_rates.get(x, "-")
)
#===============================================================================
print("\n" + "=" * 103)
print("AVERAGE PERFORMANCE ACROSS SELECTED PERIODS".center(103))
print("=" * 103)
print(summary)


### ==============================================================================================================
### PLOTS
sharpe_dyn = final_df.groupby(["Period", "Strategy"])["Sharpe"].mean().unstack()
dd_dyn = final_df.groupby(["Period", "Strategy"])["Max Drawdown"].mean().unstack()
ret_dyn = final_df.groupby(["Period", "Strategy"])["Total Return"].mean().unstack()

# insert year for axes
year_labels = [(pd.to_datetime(p) + pd.DateOffset(years=2)).strftime('%Y') for p in sharpe_dyn.index]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("GLOBAL STRATEGY ROBUSTNESS ANALYSIS (2007-2018)", fontsize=20, fontweight='bold')

# 1. Sharpe Ratio
sharpe_dyn.reset_index(drop=True).plot(ax=axes[0, 0], marker='o', markersize=5, linewidth=2)
axes[0, 0].set_title("Sharpe Ratio Dynamics", fontsize=14)

# 2. Max Drawdown
dd_dyn.reset_index(drop=True).plot(ax=axes[0, 1], marker='s', markersize=5, linewidth=2)
axes[0, 1].set_title("Max Drawdown per Period", fontsize=14)

# 3. Total Return
ret_dyn.reset_index(drop=True).plot(ax=axes[1, 0], marker='d', markersize=5, linewidth=2)
axes[1, 0].set_title("Total Return per Period", fontsize=14)

# Customization for first 3 plots
for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax.set_xticks(range(0, len(year_labels), 2))
    ax.set_xticklabels([year_labels[i] for i in range(0, len(year_labels), 2)])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlabel("Year")

# 4. Boxplot
final_df.boxplot(column='Sharpe', by='Strategy', ax=axes[1, 1], patch_artist=True)
axes[1, 1].set_title("Sharpe Distribution", fontsize=14)
plt.suptitle("")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
