import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Загружаем данные из твоего файла с результатами GARCH
with open("../../Data/Results/garch_results.pkl", "rb") as f:
    results = pickle.load(f)


def run_triple_backtest(results):
    all_metrics = []

    for r in results:
        ticker = r["summary"]["Ticker"]
        returns = r["series"]["returns"] / 100  # Доходность в долях (0.01)
        vol = r["series"]["volatility"] / 100  # Волатильность в долях (0.02)

        # --- 1. БАЗА: Buy & Hold (Всегда 1.0) ---
        weight_fixed = pd.Series(1, index=returns.index)
        ret_fixed = weight_fixed * returns

        # --- 2. TVS: Volatility Scaling (Твоя текущая) ---
        target_vol = 0.02
        weight_scaling = (target_vol / vol).clip(0, 2)
        ret_scaling = weight_scaling * returns

        # --- 3. FILTER: Volatility Filter (Новая) ---
        # Если вола выше среднего значения для этой акции — уходим в 0.
        threshold = vol.median()
        weight_filter = (vol < threshold).astype(float)
        ret_filter = weight_filter * returns

        # Собираем словарь стратегий для цикла
        strategies = {
            "Fixed (Buy&Hold)": ret_fixed,
            "Vol Scaling (TVS)": ret_scaling,
            "Vol Filter": ret_filter
        }

        # Считаем метрики для каждой
        for name, strat_ret in strategies.items():
            equity = (1 + strat_ret).cumprod()

            # Расчет показателей
            total_ret = equity.iloc[-1] - 1
            sharpe = (strat_ret.mean() / (strat_ret.std() + 1e-8)) * np.sqrt(252)
            max_dd = (equity / equity.cummax() - 1).min()

            all_metrics.append({
                "Ticker": ticker,
                "Strategy": name,
                "Total Return": total_ret,
                "Sharpe": sharpe,
                "Max Drawdown": max_dd
            })

        # --- ВИЗУАЛИЗАЦИЯ ДЛЯ ТИКЕРА ---
        plt.figure(figsize=(12, 6))
        for name, strat_ret in strategies.items():
            equity = (1 + strat_ret).cumprod()
            plt.plot(equity, label=name)

        plt.title(f"Сравнение стратегий для {ticker}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return pd.DataFrame(all_metrics)


# ЗАПУСК
results_df = run_triple_backtest(results)

# Сводная таблица (Среднее по всем акциям)
summary = results_df.groupby("Strategy")[["Total Return", "Sharpe", "Max Drawdown"]].mean()

print("\n" + "=" * 50)
print("СРЕДНИЕ РЕЗУЛЬТАТЫ ПО ВСЕМ ТИКЕРАМ")
print("=" * 50)
print(summary.round(4))