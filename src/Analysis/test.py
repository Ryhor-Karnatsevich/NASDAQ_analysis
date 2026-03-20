import pandas as pd

is_test = True
if is_test:
    path = r"../../Data/test_data_cleaned.csv"
    output_path = r"../../Data/test_analysis.csv"
else:
    path = r"../../Data/all_stocks_cleaned.csv"
    output_path = r"../../Data/all_stocks_analysis.csv"

df = pd.read_csv(path)

print

