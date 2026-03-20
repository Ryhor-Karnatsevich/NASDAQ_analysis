import pandas as pd
import glob
import os

input_path = r"../../Data/dataset/stocks/*.csv"
output_file = "../../Data/all_stocks.csv"

if not os.path.exists("../../Data"):
    os.makedirs("../../Data")


all_files = glob.glob(input_path)


first_file = pd.read_csv(all_files[0])
first_file["Ticker"] = os.path.basename(all_files[0].replace(".csv",""))
first_file.to_csv(output_file, index=False)


for file in all_files[1:]:

    df = pd.read_csv(file)
    file_name = os.path.basename(file)
    ticker = file_name.replace(".csv", "")
    df["Ticker"] = ticker

    df.to_csv(output_file, mode='a', index=False, header=False)