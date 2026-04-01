# Stock Market Data Analysis (1962-2020)

## Project Overview

This project focused on the statistical and econometric analysis of stock market data.
The goal is to examine return behavior, volatility patterns, and relationships between trading activity and price dynamics.


## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
---

## Dataset

SOURCE: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

- Historical stock data (OHLCV)
- All NASDAQ tickers
- Time period: 1962-2021

---

## Data Integration
- Raw data was merged into one csv file
- Also created test dataset with date filtration starting from 2019 to speed up and simplify the process. 

## Data Cleaning

The dataset was preprocessed to ensure data quality

### Date
- Standardized date column

### Filtration
- After reviewing the data, I found that the stocks had different listing dates. 
- I decided to shorten the sample period so that it starts from 2000-01-01 to ensure data consistency across the modern market era.

### Missing Values
- Removed 462 rows with missing values

### Duplicates
- All duplicate rows were removed

### Logical Consistency Check
- Removed invalid rows where:
  - High < Low
  - High < Close
  - Low > Close
  - Volume < 0

### Invalid Prices
- A significant number of records had invalid Open prices equal to zero.
- It has been treated as missing values and removed to ensure data integrity.

### Conclusion:
- At the end dataset contains 18371813 records.
- It is consistent and ready to be analyzed.
---

## Feature Engineering

The following variables were calculated:

- Returns 
- Log Returns
- Rolling Standard Deviation of Returns
- Moving Averages (SMA 10, SMA 50)
- Momentum (10-day return)
- Volume change
- Target (variable was defined as the next-day return to support predictive modeling) 

---

# Exploratory Data Analysis (EDA)

Two levels of analysis were defined: market level and stock level.

## Market Level

### Records Distribution
- Distribution is strongly left-skewed.
- A large proportion of stocks have observations covering nearly the entire time period.
- However, some stocks have significantly fewer data points due to later listing dates.

### Returns Distribution
- Returns are strongly centered around zero, indicating that most daily price changes are small.
- The distribution is centered around zero but deviates from normality due to the presence of fat tails.
- The distribution appears approximately symmetric.

### Volatility Distribution
- Volatility is strongly right-skewed.
- Most observations correspond to low-volatility periods, while a smaller number of observations represent higher volatility levels.

### Correlation Structure
- Stocks exhibit moderate positive correlation (around 0.3 on average).
- However, correlations are far from perfect, suggesting the presence of common market factors.

## Stock Level

### Price Behavior
- Most stocks exhibit long-term trends combined with short-term fluctuations.
- While many stocks show upward trends, some display significant declines or unstable behavior.

### Returns Behavior
- Returns fluctuate around zero and appear largely random across different stocks.

### Volatility Behavior
- Some evidence of volatility clustering is observed, where periods of higher volatility tend to persist.

### Moving Average (SMA)
- The moving average closely follows the price, smoothing short-term noise without altering the overall trend.

### Autocorrelation (ACF)
- Autocorrelation is generally close to zero, with only small deviations at a few lags.
- This suggests weak predictability of returns.

## Data Structure
- Most stocks have a large number of observations close to the maximum available.
- However, some stocks have significantly fewer data points due to later listing dates.
- To ensure data reliability, stocks with fewer than 500 observations were removed.


# Model Creating

- The goal of this section is to test the Efficient Market Hypothesis and examine whether stock behavior is predictable.

## Ordinary Least Squares (OLS Model)

The purpose of this section is to test whether simple linear relationships can predict future stock returns.

Three models were estimated using OLS with robust standard errors '**HC3**'.

**Method**
- Data was sorted by ticker and date.
- Removed stocks with less than 500 records to avoid data distortion and errors.
- Created lagged variables: Return_lag1 and Volume_lag1
- Removed missing values generated after creating lagged variables
- Train/Test split:
   - Train: before 2019
   - Test: starting with 2019
- Used robust standard errors (HC3)

### Model 1 - Lagged Returns
Return(t) = β0 + β1 * Return(t-1) + ε

(This model tests whether past returns can predict future returns)

---

![ols1](Pictures/model_1.png)

---

**Interpretation**:
- The coefficient is not statistically significant and very close to zero. 
- This indicates that past returns do not predict future returns.
- No evidence of momentum
- No evidence of mean reversion

### Model 2 - Lagged Volume
Return(t) = β0 + β1 * Volume(t-1) + ε

(This model tests whether volume change can predict future returns)

---

![ols1](Pictures/model_2.png)

---

**Interpretation**:
- Volume change does not provide predictive power for returns.

### Model 3 – Combined Model
Return(t) = β0 + β1 * Return(t-1) + β2 * Volume(t-1) + ε

(This model tests whether both return and volume change can predict future returns)

---

![ols1](Pictures/model_3.png)

---

**Interpretation**:
- Combining two variables didn't improve the model.
- Model can't predict future returns.

## Conclusion:
All three models provide consistent results:

 - Near-zero explanatory power.
 - No statistically significant predictors.
 - No improvement across different model specifications.

**Financial Insight**:
- The results support the weak form of the Efficient Market Hypothesis (historical price and volume information cannot be used to predict future returns).
- It is unlikely to build a profitable strategy based on this model.


## ARIMA model

The purpose of this section is to test whether time series structure in returns can predict future returns.

### Model specification:
   ARIMA (1,0,0)
   - AR(1): LAG RETURNS
   - d = 0: variable is already stationary
   - MA = 0: no moving averages

**Method**
- Data was sorted by ticker and date.
- Removed stocks with less than 1000 records to ensure model stability.
- Train/Test split:
   - Train: before 2019
   - Test: starting with 2019
- Model was estimated separately for each stock.
- Random sample of 10 tickers was used.

---

![ARIMA_returns](Pictures/ARIMA_returns.png)

---

**Interpretation**:
- Most coefficients are statistically significant (p < 0.05), but their values are very small.
- This means past returns have only a weak effect on future returns.
- Directional accuracy is around 0.5, so the model predicts almost like random guessing.
- Results also vary across stocks, indicating low stability.


**Financial Insight**:
- The results support the weak form of the Efficient Market Hypothesis
- It is unlikely to build a profitable strategy based on this model.


## GARCH Model

The purpose of this section is to model and predict volatility of stock returns.

Unlike previous models, this approach focuses on volatility instead of returns.

**Hypothesis:**

Stock return volatility exhibits clustering and can be modeled using GARCH-type models.

### GRID COMPARISON
GRID mode was used to compare different volatility models and select the most suitable specification.

**Models tested:**
 - GARCH
 - EGARCH
 - APARCH

**Parameters tested:**
  - (1,1)
  - (1,2)
  - (2,1)

In total, 9 model configurations were evaluated for each stock.

**Results:**
- EGARCH(2,1) was most frequently selected (13 out of 20).
- Selection was based on AIC metric.
- Other models occasionally performed better, but differences were small
- GRID results have been saved as GRID.csv

**Conclusion from GRID Research:**
- EGARCH models are better for asymmetric volatility. 
- EGARCH(2,1) was chosen as the default model for **"FINAL"** analysis

### FINAL Analysis: EGARCH(2,1)
**Method:**
- Data was sorted by ticker and date.
- Stocks with fewer than 500 observations were removed.
- Returns were scaled and clipped to ensure stability.
- Train/Test split:
   - Train: before 2019
   - Test: starting with 2019
- Model was estimated separately for each stock.
- Random sample of 100 tickers was used.
- Models with convergence issues were excluded.
- Parallel computation was used to reduce execution time and improve efficiency.

### Evaluation Metrics:
- MAE - Mean absolute error, the lower - the better.
- Relative MAE - normalized MAE for comparison across stocks.

---

![average](Pictures/average.png)

---

### Model Characteristics:
- **Alpha**: Reaction to recent shocks.
- **Beta**: Persistence of past volatility.
- **Persistence**: The sum of these effects. If it's close to 1, the "memory" of a crisis stays for a long time.
  **Results**
- High **Beta** (~0.97–0.99) indicates that volatility is driven more by past volatility than by new shocks.
- Persistence is very high ~ 0.98, indicating that volatility has a long term memory.

### Interpretation:
- The model tends to overestimate volatility, which is consistent with high persistence.
- **Mean Relative MAE** ~ 0.97, meaning that the prediction error is close to the average magnitude of volatility itself.
- This is comparable to a naive baseline where volatility is assumed to remain constant.
- Its predictive performance remains moderate and may not provide a strong advantage:
    - Some stocks have R_MAE ~ 0.65, and it is a good level of prediction and can be used.
    - Others have it ~ 4.6, indicating that model is completely useless for those stocks.
    - About 80% of stocks have R_MAE < 1, indicating that for most stocks the model performs better than a naive prediction.
- **MAE** provides an interpretable measure of prediction error in absolute terms for individual stocks.

---

![MLPI](Pictures/MLPI.png)

---

### Visual Analysis
- Predicted volatility is smoother than Absolute Returns
- Model reacts to spikes with delay
- Predictions often stay elevated after return decreases

**This results in:**
- Lagging response to market shocks
- Systematic overestimation in low-volatility periods

### Optimization
- During the programming I explored the way to reduce run time and included paralleled calculating.
- I reduced time on about 5%. From 139.22 to 132.61 seconds on a test setup.

### Conclusion
- FINAL results have been saved as FINAL.csv
- EGARCH(2,1) provides stable and consistent results across different stocks.
- Volatility clustering observed during EDA is confirmed by the model results.
- The model reacts to volatility shocks rather than predicting them in advance.
- The model successfully captures persistence in volatility.
- However, predictive performance is not consistent across all stocks:
    - For many stocks, the model performs at a baseline level.
    - For some stocks, it provides significantly better predictions (low R_MAE).
    - For others, it fails to capture volatility dynamics.

- This suggests that the model can be useful, but only for specific assets.

### Financial Insight
- GARCH-type models can be useful for risk management, portfolio allocation, and volatility-based strategies.
- The model is not suitable for predicting sudden volatility spikes, but rather for estimating the general level of risk.

### Limitations
- Model tends to overestimate volatility
- Performance slightly varies across different stocks
- Only a subset of stocks was used due to computational constraints


## Backtest

The goal of that part is to implement EGARCH(2,1) model into four strategies, choose the best one for further improving. 

### Strategies Comparison

**Method**
- Gets trained models for chosen list of stock from last iteration in "GARCH" script.
- Tickers list =  [AAPL, GILD, GOOGL, MSFT, AMZN, MLPI, PEP, COST, CSCO, AMGN]
- Selected stocks are liquid large-cap equities with sufficient historical data and low Relative MAE: (0.65–0.86).
- Evaluates 5 strategies:
    - [1] "Buy & Hold" as a baseline for other strategies.
    - [2] "Target Volatility Scaling (TVS)". Uses constant volatility equal 2% to change size of the position.
    - [3] "Volatility Filter". Has two conditions - in/out of the market. Effected by MA_50 of volatility.
    - [4] "Volatility Ratio". Uses MA_20 of volatility as a target to change size of the position. (dynamic TVS)
    - [5] "TVS + Momentum Filter". Combines basic TVS and 20 days momentum of returns as a Filter with two conditions.
- Used no leverage to make equal conditions. 

**Evaluation Metrics**
- Matrix:
    - Correlation matrix of used stocks to check if they are not too tight connected. Returns used as a values.
    - Average Correlation - average values of all values excluding diagonal ones.
  - Average Performance of Strategies:
      - Total Return - basic metric to grade the efficiency of strategy.
      - Sharpe - return/risk
      - Max Drawdown - shows how strategy managed with declines.
      - Annual Volatility - shows how strategy reduces overall volatility.
      - Hit Ratio - probability for strategy to have profitable daily return.
      - Outperformance (vs B&H) - (Sharp of strategy > Sharpe of baseline).

**Graphics**
- For each ticker, the backtest generates four comprehensive plots:
    - Equity Curve: Visualizes cumulative returns and overall capital appreciation over the testing period.
    - Drawdown Analysis: Evaluates risk resilience by showing how each strategy navigates and recovers from market declines.
    - Dynamic Exposure (Position Sizing): Displays the evolving leverage for continuous scaling strategies (2 & 4). Note: Binary filters (3 & 5) are excluded from this plot due to their 0/1 step-function nature.
    - Volatility Diagnostics: Compares GARCH-predicted volatility (red) against realized absolute returns (gray) to assess model accuracy and responsiveness.

---

![MLPI](Pictures/strategies comparison.png)

---

### Interpretation
- Matrix shows that stocks have median correlation. They don't behave like one, but have same trend.  
- That means further strategies comparison can be proceeded. 
- Strategies:
  - "TVS + Momentum Filter" achieved the highest Sharpe ratio (0.71) and lowest drawdown (-14%), indicating strong risk-adjusted performance.
  - However, it outperforms Buy & Hold only in 50% of cases, suggesting instability across assets.
  - Volatility Ratio (MA20) shows the most consistent performance (60% outperformance), acting as a stable dynamic allocation strategy.
  - Target Volatility Scaling (TVS) reduces volatility and slightly improves return.
  - Volatility Filter (MA50) significantly reduces drawdowns but sacrifices returns.

---

![MLPI](Pictures/CSCO_strategies.png)

---

### Visual Analysis
- Equity curves show that volatility-based strategies generally reduce drawdown compared to Buy & Hold.
- TVS + Momentum Filter produces smoother performance, but may lag during strong upward trends.
- Volatility Filter avoids large losses but often remains out of the market for extended periods.
- Dynamic position sizing (TVS, Volatility Ratio) adjusts exposure effectively during high volatility regimes.
- EGARCH volatility forecasts capture volatility clustering and react to market stress periods.


### Conclusion
- Volatility-based strategies improve risk-adjusted performance compared to Buy & Hold.
- The combination of volatility targeting and momentum filtering consistently delivers superior results across assets.
- TVS + Momentum Filter achieves the highest Sharpe ratio and strong return while maintaining controlled drawdowns.
- It also demonstrates stable outperformance across the majority of tickers.
- Based on these results, **"TVS + Momentum"** Filter is selected for further development and refinement in the next section.





### Backtest
2. Choose best one and Run it, try different variations and choose best parameters.
3. Robustness: different periods, stocks. and sensivity. 
4. portfolio maybe.
0. already have chosen 10 popular stocks.