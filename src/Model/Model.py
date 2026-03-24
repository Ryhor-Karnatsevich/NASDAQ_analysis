import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

is_test = False

if is_test:
    path = r"../../Data/Test Data/test_analysis.csv"
else:
    path = r"../../Data/Main Data/all_stocks_analysis.csv"

df = pd.read_csv(path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"])
# Filter new stocks
min_obs = 500
df = df.groupby("Ticker").filter(lambda x: len(x) >= min_obs)


df["Return_lag1"] = df.groupby("Ticker")["Returns"].shift(1)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Return_lag1", "Returns"])


# TRAIN / TEST SPLIT
train = df[df["Date"] < "2019-01-01"]
test  = df[df["Date"] >= "2019-01-01"]
print("Train size:", len(train))
print("Test size:", len(test))


# MODEL (OLS)
# β0 + β1 * Return_(t-1) + ε
X_train = train[["Return_lag1"]]
y_train = train["Returns"]
X_train = sm.add_constant(X_train)

model = sm.OLS(y_train, X_train).fit(cov_type='HC3')
print(model.summary())

# PREDICTIONS
X_test = test[["Return_lag1"]]
X_test = sm.add_constant(X_test)

test["pred"] = model.predict(X_test)


#---------------------------------------------------------------------
# Directional accuracy
accuracy = np.mean(
    np.sign(test["pred"]) == np.sign(test["Returns"])
)
print("Directional accuracy:", accuracy)

# MSE
mse = mean_squared_error(test["Returns"], test["pred"])
print("MSE:", mse)
























