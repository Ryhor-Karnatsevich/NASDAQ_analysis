# Stock Market Data Analysis (1999-2021)

## Project Overview

This project focused on the statistical and econometric analysis of stock market data.
The goal is to examine return behavior, volatility patters, and relationships between trading activity and price dynamics.

---

## Dataset

SOURCE: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

- Historical stock data (OHLCV)
- All NASDAQ tickers
- Time period: 1999-2021

---

## Data Cleaning

The dataset wa preprocessed to ensure data quality

### Date
- Standardized date column

### Missing Values
- Removed 649 rows with missing values

### Duplicates
- All duplicate rows were removed

### Logical Consistency Check
- Removed 1264 invalid rows where:
  - High < Low
  - High < Close
  - Low > Close
  - Volume < 0

### Invalid Prices
- A significant number of records (~1 mln) had invalid Open prices equal to zero.
- It has been treated as missing values and removed to ensure data integrity.

---

## Feature Engineering

The following variables were calculated:

- Returns 
- Log Returns
- Volatility in 10 days periods
- Moving Averages (SMA 10, SMA 50)
- Momentum (10-day return)
- Volume change
- Target (variable was defined as the next-day return to support predictive modeling) 

---

## Exploratory Data Analysis (EDA)

### Stock Graphics
- Defined 'graphics' function to make initial analysis of stocks behavior.
- Created 50 graphics and only one was corrupted. Trend was flat and abrupt. 
From that I can conclude that in general data was usable for next analysis. 

### Stock 
