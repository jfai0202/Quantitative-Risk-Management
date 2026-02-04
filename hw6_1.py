import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "testfiles", "data")

input_file = os.path.join(data_dir, "test6.csv")
output_file = os.path.join(data_dir, "my_testout_6.1.csv")

df = pd.read_csv(input_file)

# compute arithmetic returns for price columns
prices = df.iloc[:, 1:]
rets = prices.pct_change().iloc[1:]

rets.insert(0, df.columns[0], df.iloc[1:, 0].values)

rets.to_csv(output_file, index=False)