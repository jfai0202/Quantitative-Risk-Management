import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "testfiles", "data")

input_file = os.path.join(data_dir, "test1.csv")
output_file = os.path.join(data_dir, "my_testout_1.3.csv")

df = pd.read_csv(input_file)

# compute covariance using pairwise complete observations
cov_matrix = df.cov()

# save result (keep column names, no extra row index formatting)
cov_matrix.to_csv(output_file)