import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Window settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.4f}'.format

# Import data
with open("../../Data/Results/garch_results.pkl", "rb") as f:
    results = pickle.load(f)