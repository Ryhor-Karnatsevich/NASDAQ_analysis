# Stock Market Data Analysis (1962-2021)

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

- In that part I wanted to confirm or refute hypotheses of efficient market and examine valuable information I can use for building investment strategy.


### Ordinary Least Squares (OLS Model)
========================================================
- An OLS regression model was used to examine whether past returns and trading volume can predict future returns.
---
- The results show that lagged returns are statistically significant but economically weak, suggesting only minor short-term momentum effects.
- Trading volume does not appear to have a statistically significant impact on future returns.
- The overall explanatory power of the model is very low (R² ≈ 0.001), indicating that stock returns are largely unpredictable.
- Model had an error with heteroscedasticity data, so I used HC3 type to fix it.
===========================================================

#OLS model
#return




