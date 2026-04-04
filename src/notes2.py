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