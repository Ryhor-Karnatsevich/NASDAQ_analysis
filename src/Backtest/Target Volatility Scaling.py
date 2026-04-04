import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import polars as pl

# Window settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Import trained EGARCH(2,1)
with open("../../Data/Results/garch_results.pkl", "rb") as f:
    results = pickle.load(f)

# Extract unique periods and create mapping
periods = sorted(set(r["summary"]["Period"] for r in results))
p = len(periods)
print(f"Periods found: {p}")
target_periods = ['2007-06-01','2015-06-01','2018-06-01']

ticker_list = list(set(r["summary"]["Ticker"] for r in results))

# =====================================================
# Fast csv import and filtration to add "Close" column
path = r"../../Data/Main Data/all_stocks_analysis.csv"
df = (
    pl.scan_csv(path)
    .select(["Date", "Ticker", "Close"])
    .filter(
        (pl.col("Ticker").is_in(ticker_list)) &
        (pl.col("Date") > "2006-06-01")
    )
    .collect()
)
df = df.to_pandas()
prices_pivot = df.pivot(index="Date", columns="Ticker", values="Close")
prices_pivot.index = pd.to_datetime(prices_pivot.index)
# =====================================================

def strategies_backtest(results, period_filter=None, verbose=False,
                        rebalance=0.035, vol_discount=0.9):

    all_metrics = []
    diagnostic_positions = [] # for sensitivity plot

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
        turnover_fixed = 0
        target = 0

        # [2] "TVS with transaction and margin costs"
        # Costs Setup
        target_0 = 0.02
        margin_annual_rate_0 = 0.05  # 5% annually for leverage
        margin_daily_rate_0 = margin_annual_rate_0 / 252
        comm_rate_0 = 0.0005  # 0.05% - transaction cost
        # Raw return
        position_basic = (target_0 / volatility).clip(0, 2)
        return_raw_basic = position_basic * returns
        # Transaction Costs and turnover
        trades_basic = position_basic.diff().abs().fillna(0)
        turnover_basic = trades_basic.mean() * 252
        commissions_0 = trades_basic * comm_rate_0
        # Margin Costs
        leverage_used_0 = (position_basic - 1).clip(lower=0)
        margin_costs_0 = leverage_used_0 * margin_daily_rate_0
        # Final Return
        return_net_basic = return_raw_basic - commissions_0 - margin_costs_0


        # ====================================================================================================
        # [3] Target Volatility Scaling Advanced
        # 1  Setup: Leverage, Costs and Prices
        leverage = 2
        # vol_discount = 0.8  # value to offset predicted volatility, if it is overrated
        # rebalance = 0.035  # value for rebalancing threshold
        comm_rate = 0.0005
        margin_daily_rate = 0.05 / 252  # annual rate / trading days
        ticker_prices = prices_pivot[ticker].reindex(returns.index)
        volatility = volatility.clip(lower=0.0125)  # limit flour for volatility

        # 2  Asymmetric Target (based on sma_200)
        volatility_target = 0.02
        # momentum = ticker_prices.pct_change(20).shift(1)
        # volatility_target = np.where(momentum > 0, 0.02, 0.01)

        # 3  Volatility Risk Premium Offset
        raw_position = (volatility_target / (volatility * vol_discount)).clip(0, leverage)

        # 4  Rebalancing Threshold (Cutting unnecessary turnover & slippage)
        position = raw_position.copy()
        for i in range(1, len(position)):
            # Only update position if the change is greater than 5%
            if abs(raw_position.iloc[i] - position.iloc[i - 1]) < rebalance:
                position.iloc[i] = position.iloc[i - 1]

        diagnostic_positions.extend(position.values) # for sensitivity plot

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
        turnover = trades.mean() * 252

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
            "Buy & Hold": (return_fixed, weight_fixed,turnover_fixed,None, None),
            "TVS with transaction and margin costs": (return_net_basic, position_basic,turnover_basic,target_0,return_fixed),
            "Target Volatility Scaling (TVS)": (return_net, position,turnover,volatility_target,return_fixed)
        }

        # Another subiteration to add values to final table and to plot dictionary
        for name, (strategy_return, weight,turnover,target,return_fixed) in strategies.items():
            equity = (1 + strategy_return).cumprod()  # creates equity curve (line of returns of all previous time)
            metrics = calculate_metrics(strategy_return, equity, turnover, target, return_fixed)  # Use function that below

            if metrics:
                # Adding data to list for final table
                all_metrics.append({
                    "Ticker": ticker,
                    "Period": period,
                    "Strategy": name,
                    **metrics # from function "calculate_metrics"
                })

    return pd.DataFrame(all_metrics), diagnostic_positions


# Function to calculate metrics
# ----------------------------------------------------------------------------------------------------------------------------------
def calculate_metrics(returns_series, equity_series, turnover,target,return_fixed ):
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

    # CVAR
    percentile = 0.05
    var = np.percentile(returns_series, percentile * 100)
    cvar = returns_series[returns_series <= var].mean()

    # Volatility Target Deviation
    if target is not None:
        realized_vol = returns_series.rolling(20).std()
        if isinstance(target, pd.Series):
            target_aligned = target.reindex(returns_series.index)
        else:
            target_aligned = pd.Series(target, index=returns_series.index)
        vol_target_dev = np.nanmean(np.abs(realized_vol - target_aligned))
    else:
        vol_target_dev = np.nan

    # Tail Risk Reduction
    if return_fixed is not None:
        bh_var = np.percentile(return_fixed, 5)
        bh_cvar = return_fixed[return_fixed <= bh_var].mean()
        if np.isnan(bh_cvar) or bh_cvar == 0:
            tail_risk_reduction = np.nan
        else:
            tail_risk_reduction = (cvar - bh_cvar) / abs(bh_cvar)
    else:
        tail_risk_reduction = np.nan


    # That dictionary appends to ticker dictionary in all_metrics list
    return {
        "Total Return": total_return,
        "Sharpe": sharpe,
        "Max_Drawdown": max_dd,
        "Annual_Vol": annual_vol,
        "Hit_Ratio": hit_ratio,
        "Turnover": turnover,
        "CVaR": cvar,
        "RMSE_Vol": vol_target_dev,
        "TRR": tail_risk_reduction
    }



# =========================================================
# EXECUTION
# =========================================================

all_results = []

rebalance_values = [0.02, 0.03, 0.035, 0.05]
vol_discount_values = [0.6, 0.8, 1.0]

sensitivity_results = []
all_positions = []
main_positions = []  # for basic leverage plot

for period in periods:

    print("\n" + "="*60)
    print(f"Processing Period: {period}")
    print("="*60)

    filtered_results = [r for r in results if r["summary"]["Period"] == period]

    results_df, period_positions = strategies_backtest(filtered_results)
    results_df["Period"] = period
    all_results.append(results_df)
    all_positions.extend(period_positions) # for plots
    main_positions.extend(period_positions)

    for reb in rebalance_values:
        for vd in vol_discount_values:

            tmp_df, _ = strategies_backtest(
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
# PERIODS ANALYSIS
# =========================================================

regime_df = final_df[final_df["Period"].isin(target_periods)].copy()
regime_df["Periods"] = regime_df["Period"].map({
    target_periods[0]: "2007-2009",
    target_periods[1]: "2015-2017",
    target_periods[2]: "2018-2020"
}).fillna("Other")

regime_summary = regime_df.groupby(["Periods", "Strategy"]).agg({
    "Total Return": "mean",
    "Sharpe": "mean",
    "Max_Drawdown": "mean",
    "Annual_Vol": "mean",
    "Hit_Ratio": "mean",
    "Turnover": "mean",
    "CVaR": "mean",
    "RMSE_Vol": "mean",
    "TRR": "mean"
}).sort_values(by=["Periods", "Sharpe"],ascending=[True, False])

# Adding Outperformance column
pivot_regime = regime_df.pivot_table(
    index=['Periods', 'Ticker'],
    columns='Strategy',
    values='Sharpe'
)
win_rates_regime = {}
for (period_name, strategy), _ in regime_summary.iterrows():
    if strategy != "Buy & Hold":
        current_period_data = pivot_regime.xs(period_name, level='Periods')
        wins = (current_period_data[strategy] > current_period_data["Buy & Hold"]).sum()
        total = len(current_period_data)
        win_rates_regime[(period_name, strategy)] = f"{wins}/{total} ({wins / total:.1%})"

regime_summary["Outperf. (vs B&H)"] = [
    win_rates_regime.get(index, "-") for index in regime_summary.index
]

# =========================================================
# FINAL SUMMARY
summary = final_df.groupby("Strategy").agg({
    "Total Return": "mean",
    "Sharpe": "mean",
    "Max_Drawdown": "mean",
    "Annual_Vol": "mean",
    "Hit_Ratio": "mean",
    "Turnover": "mean",
    "CVaR": "mean",
    "RMSE_Vol": "mean",
    "TRR": "mean"
}).sort_values("Sharpe", ascending=False)

# Periods printing setup
regime_print = regime_summary.copy()
cols_2_decimal = ["Turnover", "Max_Drawdown", "Annual_Vol"]
cols_4_decimal = ["Sharpe", "Total Return","Hit_Ratio", "CVaR", "RMSE_Vol", "TRR"]
for col in cols_2_decimal:
    if col in regime_print.columns:
        regime_print[col] = regime_print[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
for col in cols_4_decimal:
    if col in regime_print.columns:
        regime_print[col] = regime_print[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
regime_print = regime_print.fillna("-")

# FINAL table printing setup
summary_print = summary.copy()
for col in cols_2_decimal:
    if col in summary_print.columns:
        summary_print[col] = summary_print[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
for col in cols_4_decimal:
    if col in summary_print.columns:
        summary_print[col] = summary_print[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
summary_print = summary_print.fillna("-")

# printing "3 PERIODS ANALYSIS"  and  "FINAL SUMMARY"
def print_styled_table(df, title):
    print("\n" + "=" * 170)
    print(title.center(170))
    print("=" * 170)
    output = df.to_string(justify='center', col_space=10)
    print(output)
    print("-" * 170)
print_styled_table(regime_print, "3 PERIODS ANALYSIS")
print_styled_table(summary_print, "AVERAGE PERFORMANCE ACROSS SELECTED PERIODS")


### ==============================================================================================================
### PLOTS FOR PERIODS
### ==============================================================================================================


# ===================================================================================================
# CORE ANALYSIS (4 PLOTS)
# Sharpe, Max Drawdown, Return, Equity Curve
sharpe_dyn = final_df.groupby(["Period", "Strategy"])["Sharpe"].mean().unstack()
dd_dyn = final_df.groupby(["Period", "Strategy"])["Max_Drawdown"].mean().unstack()
ret_dyn = final_df.groupby(["Period", "Strategy"])["Total Return"].mean().unstack()

# Equity Curve Calculation (Average Growth of $1)
equity_data = {}
for name in ["Buy & Hold", "Target Volatility Scaling (TVS)", "TVS with transaction and margin costs"]:
    avg_ret_per_period = final_df[final_df["Strategy"] == name].groupby("Period")["Total Return"].mean()
    equity_data[name] = (1 + avg_ret_per_period).cumprod()
equity_dyn = pd.DataFrame(equity_data)

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

# 4. Equity Curve
equity_dyn.reset_index(drop=True).plot(ax=axes[1, 1], linewidth=3)
axes[1, 1].set_title("Cumulative Growth of $1 (Average)", fontsize=14)

# Customization for all 4 plots
for ax in axes.flat:
    ax.set_xticks(range(0, len(year_labels)))
    ax.set_xticklabels(year_labels, rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlabel("Year")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# ===================================================================================================



#===================================================================================================
# Volatility Targeting Accuracy
# Aggregated portfolio volatility vs target
all_vols = []
for r in results:
    s_ret = r["series"]["returns"] / 100
    r_vol = s_ret.rolling(20).std() * np.sqrt(252)
    all_vols.append(r_vol)

mean_realized_vol = pd.concat(all_vols, axis=1).mean(axis=1).iloc[:500]

plt.figure(figsize=(12, 6))
plt.plot(mean_realized_vol.values, label='Average Realized Volatility (Portfolio)', color='orange', linewidth=2)
plt.axhline(y=0.02, color='black', linestyle='--', label='Target Volatility (2%)')
plt.fill_between(range(len(mean_realized_vol)), mean_realized_vol.values, 0.02, color='gray', alpha=0.1)
plt.title("Volatility Targeting Accuracy (Portfolio Level)", fontsize=14)
plt.legend()
plt.grid(alpha=0.2)
plt.show()
#===================================================================================================



#===================================================================================================
# Leverage Distribution (Main Strategy Only)
plt.figure(figsize=(10, 6))
plt.hist(main_positions, bins=50, color='teal', alpha=0.7, edgecolor='white')
plt.axvline(x=1.0, color='red', linestyle='--', label='No Leverage (100% Weight)')
plt.title("Leverage Usage Distribution (Default Parameters)", fontsize=14)
plt.xlabel("Position Size (0.0 to 2.0)")
plt.ylabel("Frequency (Days)")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
#===================================================================================================




### ==============================================================================================================
### SENSITIVITY PLOTS
### ==============================================================================================================


# ==============================================================================================================
# ROBUSTNESS: SENSITIVITY ANALYSIS (4 HEATMAPS)
# Sharpe, CVaR, Return, Std Sharpe
tvs_sens = sensitivity_df[sensitivity_df["Strategy"] == "Target Volatility Scaling (TVS)"]

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle("SENSITIVITY ANALYSIS: RISK & PERFORMANCE HEATMAPS", fontsize=22, fontweight='bold')

metrics = [
    ("Sharpe", "Mean Sharpe Ratio (Efficiency)", "YlGnBu"),
    ("CVaR", "Mean CVaR (Tail Risk)", "Reds_r"),
    ("Total Return", "Mean Total Return (Performance)", "Greens"),
    ("Sharpe", "Std Dev of Sharpe (Stability)", "Purples")
]

for i, (metric, title, cmap) in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    if "Std Dev" in title:
        pivot_data = tvs_sens.groupby(["rebalance", "vol_discount"])[metric].std().unstack()
    else:
        pivot_data = tvs_sens.groupby(["rebalance", "vol_discount"])[metric].mean().unstack()

    fmt = ".2%" if metric == "Total Return" else ".3f"

    sns.heatmap(pivot_data, annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar_kws={'label': metric})

    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Rebalance Threshold")
    ax.set_xlabel("Volatility Discount")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# ==============================================================================================================



#===================================================================================================
# SMA Window Sensitivity
sma_values = [50, 100, 150, 200, 250]
sma_results = []

for sma in sma_values:
    res_df, _ = strategies_backtest(filtered_results, sma_window=sma)
    res_df["sma_window"] = sma
    sma_results.append(res_df)

sma_df = pd.concat(sma_results)
sma_plot = sma_df.groupby("sma_window")["Sharpe"].mean()

plt.figure(figsize=(10, 5))
sma_plot.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Sensitivity to SMA Window (Trend Detection)", fontsize=14)
plt.ylabel("Average Sharpe Ratio")
plt.xlabel("SMA Lookback Period")
plt.grid(axis='y', alpha=0.3)
plt.show()
#===================================================================================================



#===================================================================================================
# Leverage Sensitivity
plt.figure(figsize=(10, 6))
plt.hist(all_positions, bins=30, color='teal', alpha=0.7, edgecolor='white')
plt.axvline(x=1.0, color='red', linestyle='--', label='No Leverage Threshold')
plt.title("Leverage Usage Distribution", fontsize=14)
plt.xlabel("Position Size (Leverage)")
plt.ylabel("Frequency (Days)")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
#===================================================================================================