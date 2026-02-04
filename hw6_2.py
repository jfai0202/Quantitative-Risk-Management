import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "testfiles", "data")

input_file = os.path.join(data_dir, "test6.csv")
output_file = os.path.join(data_dir, "my_testout_6.2.csv")

df = pd.read_csv(input_file)

# compute log returns for price columns
prices = df.iloc[:, 1:]
log_rets = np.log(prices / prices.shift(1)).iloc[1:]

log_rets.insert(0, df.columns[0], df.iloc[1:, 0].values)

log_rets.to_csv(output_file, index=False)