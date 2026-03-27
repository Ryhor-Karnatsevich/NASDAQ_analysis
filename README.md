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


